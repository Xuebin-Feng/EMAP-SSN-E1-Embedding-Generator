# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% Import Necessary Libraries
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
import sys
import numpy as np
import h5py
import torch
from tqdm import tqdm
import re
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

# Import E1 Model Components
try:
    from E1.batch_preparer import E1BatchPreparer
    from E1.modeling import E1ForMaskedLM
    HAS_E1 = True
except ImportError:
    HAS_E1 = False

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_SET = "Foldtype_IV_ATAs_13000"
NETWORK_MODEL_NAME = "esmc_6b"  # Model used to generate the network

# --- Paths ---
INPUT_FASTA = os.path.join("Sequence_Sets", f"{SEQUENCE_SET}.fasta")
if "blast" in NETWORK_MODEL_NAME.lower():
    INPUT_NETWORK_H5 = os.path.join("Networks_EValues", f"{SEQUENCE_SET}_[BLAST]_EValue.h5")
else:
    INPUT_NETWORK_H5 = os.path.join("Networks_EValues", f"{SEQUENCE_SET}_[{NETWORK_MODEL_NAME}]_network.h5")

# --- Embedding Generation Parameters ---
EMBEDDING_MODE = "RA"                  # "NORM" (Single sequence) or "RA" (Retrieval-Augmented)
BATCH_SIZE = 1                         # Keep at 1 for RA mode
E1_MODEL_MAPPINGS = {
    "E1_RA": "Profluent-Bio/E1-600m",
    "E1_NORM": "Profluent-Bio/E1-600m",
}
SAVING_MODE = "float16"                # Target precision for HDF5 save
OUTPUT_HDF5 = os.path.join("Embeddings", f"{SEQUENCE_SET}_[E1_{EMBEDDING_MODE}]_embeddings.h5")

# --- Homolog Search Parameters (For RA Mode) ---
MAX_SEQ_LENGTH = 4096                  # Filter out overly long sequences
TOP_K_SEARCH = 1000                    # Max rank to consider
SELECTION_STRIDE = 20                  # Pick every 20th homolog (20, 40, 60...)
SCORE_MODE = "global"                  # "local" or "global"
NORMALIZATION_MODE = "alignment_length"

# --- Sparse Imputation Parameters (For RA Mode with Sparse Network) ---
POOLING_METHOD = "max"                  # "mean" or "max"
LENGTH_RATIO_POWER = 1.0                # Exponent for length penalty
SIMILARITY_MODEL_NAME = None            # Embedding model name to use for cosine similarity (e.g. "esmc_600m"). If None, defaults to NETWORK_MODEL_NAME.

# --- Embedding-Only Homolog Selection Switch ---
USE_ONLY_EMBEDDINGS = False             # If True, ignore network file and select homologs using only embedding similarity

# --- Context Parameters ---
MAX_HOMOLOGS = 50                      # Exact maximum number of homologs to append to the query

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# STANDALONE FASTA SANITIZATION
# ==========================================
VALID_RESIDUE_CODES = "ACDEFGHIKLMNPQRSTVWYBZJXUO"
RESIDUE_BOUNDARY_PATTERN = re.compile(
    rf"[{VALID_RESIDUE_CODES}].*[{VALID_RESIDUE_CODES}]|[{VALID_RESIDUE_CODES}]"
)
INVALID_RESIDUE_PATTERN = re.compile(rf"[^{VALID_RESIDUE_CODES}]")


def read_fasta_records(file_path):
    """Read a FASTA file into ordered header and sequence lists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    headers = []
    sequences = []
    current_header = None
    current_sequence = []

    with open(file_path, "r", encoding="utf-8-sig") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_sequence))
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(current_sequence))

    return headers, sequences


def sanitize_header(header):
    """Apply the canonical SSN header character and whitespace rules."""
    safe_header = header.translate(str.maketrans("[]{}", "()()"))
    safe_header = re.sub(r'[?*"#%@$/\\]', "_", safe_header)
    safe_header = re.sub(r"_+", "_", safe_header)
    safe_header = re.sub(r"\s+", " ", safe_header).strip()
    return safe_header, safe_header != header


def sanitize_sequence(seq):
    """Uppercase a sequence, trim terminal artifacts, and mask internal ones."""
    upper_seq = seq.upper()
    match = RESIDUE_BOUNDARY_PATTERN.search(upper_seq)

    if not match:
        return "", upper_seq, []

    core_seq = match.group(0)
    stripped_chars = upper_seq[:match.start()] + upper_seq[match.end():]
    invalid_internal_chars = INVALID_RESIDUE_PATTERN.findall(core_seq)
    final_seq = INVALID_RESIDUE_PATTERN.sub("X", core_seq)
    return final_seq, stripped_chars, invalid_internal_chars


def select_preferred_header(current_headers):
    """Select the longest header and report true duplicate-header counts."""
    header_counts = Counter(current_headers)
    unique_headers = list(header_counts)
    duplicate_headers = {
        header for header, count in header_counts.items() if count > 1
    }
    duplicates_count = sum(count - 1 for count in header_counts.values())

    if len(unique_headers) > 1:
        best_header = sorted(
            unique_headers,
            key=lambda value: (-len(value), value),
        )[0]
        discarded_headers = sorted(
            (header for header in unique_headers if header != best_header),
            key=lambda value: (-len(value), value),
        )
    else:
        best_header = unique_headers[0]
        discarded_headers = []

    return best_header, discarded_headers, duplicate_headers, duplicates_count


def allocate_unique_headers(header_to_seqs):
    """Allocate globally unique output headers without replacing existing names."""
    reserved_headers = set(header_to_seqs)
    used_headers = set()
    assigned_by_header = {}

    for header, unique_seqs in header_to_seqs.items():
        if len(unique_seqs) == 1:
            assigned_headers = [header]
        else:
            assigned_headers = []
            suffix = 1
            separator = "" if header.endswith("_") else "_"
            for _ in unique_seqs:
                candidate = f"{header}{separator}{suffix}"
                while candidate in reserved_headers or candidate in used_headers:
                    suffix += 1
                    candidate = f"{header}{separator}{suffix}"
                assigned_headers.append(candidate)
                suffix += 1

        for assigned_header in assigned_headers:
            if assigned_header in used_headers:
                raise RuntimeError(
                    f"Unable to allocate a unique FASTA header for '{header}'."
                )
            used_headers.add(assigned_header)

        assigned_by_header[header] = assigned_headers

    return assigned_by_header


def sanitize_fasta_records(headers, sequences):
    """Sanitize and deduplicate FASTA records without filtering by name or length."""
    if len(headers) != len(sequences):
        raise ValueError("FASTA header and sequence counts do not match.")

    stats = {
        "original_records": len(headers),
        "final_records": 0,
        "headers_modified": 0,
        "sequences_modified": 0,
        "empty_sequences_removed": 0,
        "exact_duplicates_removed": 0,
        "different_headers_merged": 0,
        "headers_renamed": 0,
        "changed": False,
    }

    seq_to_headers = {}
    for header, seq in zip(headers, sequences):
        safe_header, header_modified = sanitize_header(header)
        cleaned_seq, _, _ = sanitize_sequence(seq)

        if header_modified:
            stats["headers_modified"] += 1
        if cleaned_seq != seq:
            stats["sequences_modified"] += 1

        if cleaned_seq:
            seq_to_headers.setdefault(cleaned_seq, []).append(safe_header)
        else:
            stats["empty_sequences_removed"] += 1

    header_to_seqs = {}
    for seq, current_headers in seq_to_headers.items():
        (
            best_header,
            discarded_headers,
            _,
            duplicates_count,
        ) = select_preferred_header(current_headers)

        stats["exact_duplicates_removed"] += duplicates_count
        stats["different_headers_merged"] += len(discarded_headers)
        header_to_seqs.setdefault(best_header, []).append(seq)

    assigned_headers_by_base = allocate_unique_headers(header_to_seqs)
    clean_headers = []
    clean_sequences = []

    for header, unique_seqs in header_to_seqs.items():
        assigned_headers = assigned_headers_by_base[header]
        if len(unique_seqs) > 1:
            stats["headers_renamed"] += len(unique_seqs)

        clean_headers.extend(assigned_headers)
        clean_sequences.extend(unique_seqs)

    if len(clean_headers) != len(set(clean_headers)):
        raise RuntimeError("FASTA sanitization produced duplicate output headers.")

    stats["final_records"] = len(clean_headers)
    stats["changed"] = any(
        stats[key]
        for key in (
            "headers_modified",
            "sequences_modified",
            "empty_sequences_removed",
            "exact_duplicates_removed",
            "different_headers_merged",
            "headers_renamed",
        )
    ) or stats["final_records"] != stats["original_records"]

    return clean_headers, clean_sequences, stats


def print_sanitization_result(stats):
    """Print a compact result only when sanitization changed the records."""
    if not stats["changed"]:
        return False

    print("\nFASTA sanitization result:")
    print(f"  Original records:          {stats['original_records']}")
    print(f"  Final records:             {stats['final_records']}")

    labels = (
        ("headers_modified", "Headers modified"),
        ("sequences_modified", "Sequences modified"),
        ("empty_sequences_removed", "Empty sequences removed"),
        ("exact_duplicates_removed", "Exact duplicates removed"),
        ("different_headers_merged", "Different headers merged"),
        ("headers_renamed", "Headers renamed"),
    )
    for key, label in labels:
        if stats[key]:
            print(f"  {label + ':':<27} {stats[key]}")

    return True


def load_sanitized_fasta(file_path):
    """Read, sanitize, optionally report, and return one FASTA record set."""
    headers, sequences = read_fasta_records(file_path)
    clean_headers, clean_sequences, stats = sanitize_fasta_records(
        headers,
        sequences,
    )
    print_sanitization_result(stats)
    return clean_headers, clean_sequences, stats


# ==========================================
# STANDALONE EMBEDDING HDF5 CONTRACT
# ==========================================
SAVING_MODE_DTYPES = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}
REQUIRED_ATTRIBUTES = (
    "model_name",
    "saving_mode",
    "num_sequences",
    "generation_complete",
)
REQUIRED_OBJECTS = ("headers", "sequences", "input_hashes", "embeddings")
SANITIZED_RESIDUE_CODES = frozenset(VALID_RESIDUE_CODES)


@dataclass(frozen=True)
class EmbeddingManifest:
    headers: list
    sequences: list
    input_hashes: list
    model_name: str
    saving_mode: str
    generation_complete: bool
    feature_dimension: int | None

    @property
    def sequence_by_header(self):
        return dict(zip(self.headers, self.sequences))

    @property
    def input_hash_by_header(self):
        return dict(zip(self.headers, self.input_hashes))


def _decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_string_dataset(dataset):
    """Decode one HDF5 string dataset into ordinary Python strings."""
    return [_decode_text(value) for value in dataset[:]]


def dtype_for_saving_mode(saving_mode):
    """Return the required NumPy dtype for a persisted saving mode."""
    try:
        return SAVING_MODE_DTYPES[saving_mode]
    except KeyError as exc:
        raise ValueError(
            "saving_mode must be exactly 'float16' or 'float32'."
        ) from exc


def validate_manifest_records(headers, sequences):
    """Validate the one-to-one sanitized header/sequence manifest."""
    if len(headers) != len(sequences):
        raise ValueError("Embedding headers and sequences must have equal lengths.")

    seen = set()
    for index, (header, sequence) in enumerate(zip(headers, sequences)):
        if not isinstance(header, str) or not header:
            raise ValueError(f"Embedding header at index {index} is empty or invalid.")
        if header in {".", ".."}:
            raise ValueError(f"Embedding header '{header}' is not a safe HDF5 key.")
        if "/" in header or "\\" in header or "\x00" in header:
            raise ValueError(
                f"Embedding header '{header}' contains a path or null character."
            )
        if header in seen:
            raise ValueError(f"Duplicate embedding header '{header}'.")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(
                f"Sanitized sequence for header '{header}' is empty or invalid."
            )
        invalid_codes = sorted(set(sequence) - SANITIZED_RESIDUE_CODES)
        if invalid_codes:
            raise ValueError(
                f"Sequence for header '{header}' is not sanitized; unsupported "
                f"code '{invalid_codes[0]}' remains."
            )
        seen.add(header)


def validate_input_hashes(input_hashes, expected_count):
    """Validate the ordered SHA-256 context hash manifest."""
    if len(input_hashes) != expected_count:
        raise ValueError(
            "Embedding input hashes must have the same length as the headers."
        )
    for index, input_hash in enumerate(input_hashes):
        if (
            not isinstance(input_hash, str)
            or len(input_hash) != 64
            or any(char not in "0123456789abcdef" for char in input_hash)
        ):
            raise ValueError(
                f"Embedding input hash at index {index} is not a SHA-256 digest."
            )


def validate_embedding_array(
    embedding,
    sequence,
    saving_mode,
    feature_dimension=None,
    require_finite=False,
    header=None,
):
    """Validate one generated array or HDF5 embedding dataset."""
    label = f" for '{header}'" if header is not None else ""
    if getattr(embedding, "ndim", None) != 2:
        raise ValueError(f"Embedding{label} must be two-dimensional.")
    if embedding.shape[0] != len(sequence):
        raise ValueError(
            f"Embedding{label} has {embedding.shape[0]} rows but its stored "
            f"sequence has {len(sequence)} residues."
        )
    if np.dtype(embedding.dtype) != dtype_for_saving_mode(saving_mode):
        raise ValueError(
            f"Embedding{label} has dtype {embedding.dtype}, not {saving_mode}."
        )
    current_dimension = int(embedding.shape[1])
    if current_dimension <= 0:
        raise ValueError(f"Embedding{label} has an empty feature dimension.")
    if feature_dimension is not None and current_dimension != feature_dimension:
        raise ValueError(
            f"Embedding{label} has feature dimension {current_dimension}; "
            f"expected {feature_dimension}."
        )
    if require_finite and not np.isfinite(np.asarray(embedding)).all():
        raise ValueError(f"Embedding{label} contains NaN or infinite values.")
    return current_dimension


def _validate_embedding_group(
    hf,
    headers,
    sequences,
    saving_mode,
    require_all,
):
    emb_group = hf["embeddings"]
    if not isinstance(emb_group, h5py.Group):
        raise ValueError("'/embeddings' must be an HDF5 group.")

    expected = set(headers)
    actual = set(emb_group.keys())
    orphaned = sorted(actual - expected)
    if orphaned:
        raise ValueError(
            f"Embedding database contains unexpected dataset '{orphaned[0]}'."
        )
    missing = sorted(expected - actual)
    if require_all and missing:
        raise ValueError(
            f"Embedding database is missing dataset '{missing[0]}'."
        )

    sequence_by_header = dict(zip(headers, sequences))
    feature_dimension = None
    for header in headers:
        if header not in emb_group:
            continue
        dataset = emb_group[header]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"Embedding object '{header}' is not a dataset.")
        feature_dimension = validate_embedding_array(
            dataset,
            sequence_by_header[header],
            saving_mode,
            feature_dimension=feature_dimension,
            header=header,
        )
    return feature_dimension


def read_embedding_manifest(
    hf,
    *,
    require_complete=True,
    validate_embeddings=True,
):
    """Read and validate the required standalone E1 embedding interface."""
    missing_attrs = [name for name in REQUIRED_ATTRIBUTES if name not in hf.attrs]
    if missing_attrs:
        raise ValueError(
            "Embedding file is missing required attribute(s): "
            + ", ".join(missing_attrs)
            + ". Legacy embedding files are unsupported."
        )
    missing_objects = [name for name in REQUIRED_OBJECTS if name not in hf]
    if missing_objects:
        raise ValueError(
            "Embedding file is missing required object(s): "
            + ", ".join(f"/{name}" for name in missing_objects)
            + ". Legacy embedding files are unsupported."
        )
    for name in ("headers", "sequences", "input_hashes"):
        if not isinstance(hf[name], h5py.Dataset) or hf[name].ndim != 1:
            raise ValueError(f"'/{name}' must be a one-dimensional dataset.")

    headers = decode_string_dataset(hf["headers"])
    sequences = decode_string_dataset(hf["sequences"])
    input_hashes = decode_string_dataset(hf["input_hashes"])
    validate_manifest_records(headers, sequences)
    validate_input_hashes(input_hashes, len(headers))

    model_name = _decode_text(hf.attrs["model_name"])
    if model_name not in E1_MODEL_MAPPINGS:
        raise ValueError(
            f"Embedding model '{model_name}' is not a supported E1 model label."
        )
    saving_mode = _decode_text(hf.attrs["saving_mode"])
    dtype_for_saving_mode(saving_mode)
    try:
        num_sequences = int(hf.attrs["num_sequences"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding attribute 'num_sequences' is invalid.") from exc
    if num_sequences != len(headers):
        raise ValueError(
            f"num_sequences is {num_sequences}, but /headers contains "
            f"{len(headers)} records."
        )

    complete_value = hf.attrs["generation_complete"]
    if not isinstance(complete_value, (bool, np.bool_)):
        raise ValueError("Embedding attribute 'generation_complete' must be boolean.")
    generation_complete = bool(complete_value)
    if require_complete and not generation_complete:
        raise ValueError("Embedding generation is incomplete.")

    feature_dimension = None
    if validate_embeddings:
        feature_dimension = _validate_embedding_group(
            hf,
            headers,
            sequences,
            saving_mode,
            require_all=require_complete or generation_complete,
        )

    return EmbeddingManifest(
        headers=headers,
        sequences=sequences,
        input_hashes=input_hashes,
        model_name=model_name,
        saving_mode=saving_mode,
        generation_complete=generation_complete,
        feature_dimension=feature_dimension,
    )


def write_embedding_manifest(
    hf,
    headers,
    sequences,
    input_hashes,
    model_name,
    saving_mode,
    *,
    replace=False,
):
    """Write and flush metadata before any embedding datasets are created."""
    headers = list(headers)
    sequences = list(sequences)
    input_hashes = list(input_hashes)
    validate_manifest_records(headers, sequences)
    validate_input_hashes(input_hashes, len(headers))
    dtype_for_saving_mode(saving_mode)
    if model_name not in E1_MODEL_MAPPINGS:
        raise ValueError(
            f"model_name must be one of: {', '.join(E1_MODEL_MAPPINGS)}."
        )

    existing = [
        name for name in ("headers", "sequences", "input_hashes") if name in hf
    ]
    if existing and not replace:
        raise ValueError("Embedding manifest already exists.")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    next_names = {
        "headers": "_headers_next",
        "sequences": "_sequences_next",
        "input_hashes": "_input_hashes_next",
    }
    for temporary_name in next_names.values():
        if temporary_name in hf:
            del hf[temporary_name]

    values_by_name = {
        "headers": headers,
        "sequences": sequences,
        "input_hashes": input_hashes,
    }
    for final_name, temporary_name in next_names.items():
        hf.create_dataset(
            temporary_name,
            data=np.asarray(values_by_name[final_name], dtype=object),
            dtype=string_dtype,
        )

    hf.attrs["model_name"] = model_name
    hf.attrs["saving_mode"] = saving_mode
    hf.attrs["num_sequences"] = len(headers)
    hf.attrs["generation_complete"] = False
    hf.flush()

    for final_name, temporary_name in next_names.items():
        if final_name in hf:
            del hf[final_name]
        hf.move(temporary_name, final_name)
    hf.flush()


def create_metadata_first_file(
    hf,
    headers,
    sequences,
    input_hashes,
    model_name,
    saving_mode,
):
    """Initialize a new output file in the required metadata-first order."""
    if len(hf) or len(hf.attrs):
        raise ValueError("Output HDF5 file must be empty before initialization.")
    write_embedding_manifest(
        hf,
        headers,
        sequences,
        input_hashes,
        model_name,
        saving_mode,
    )
    emb_group = hf.create_group("embeddings")
    hf.flush()
    return emb_group


def mark_generation_complete(hf):
    """Validate the finished database, mark it complete, and flush it."""
    manifest = read_embedding_manifest(
        hf,
        require_complete=False,
        validate_embeddings=True,
    )
    expected = set(manifest.headers)
    actual = set(hf["embeddings"].keys())
    if expected != actual:
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"Embedding database is missing dataset '{missing[0]}'.")
        orphaned = sorted(actual - expected)
        raise ValueError(
            f"Embedding database contains unexpected dataset '{orphaned[0]}'."
        )
    hf.attrs["generation_complete"] = True
    hf.flush()
    return read_embedding_manifest(hf, require_complete=True)


def resolve_e1_model(embedding_mode):
    """Resolve the stored E1 label and its sole supported model checkpoint."""
    normalized_mode = str(embedding_mode).upper()
    model_name = f"E1_{normalized_mode}"
    try:
        checkpoint = E1_MODEL_MAPPINGS[model_name]
    except KeyError as exc:
        raise ValueError("embedding_mode must be exactly 'RA' or 'NORM'.") from exc
    return normalized_mode, model_name, checkpoint


def build_input_hash(checkpoint, embedding_mode, homolog_sequences, query_sequence):
    """Hash the exact ordered biological context used to generate one embedding."""
    payload = {
        "checkpoint": checkpoint,
        "embedding_mode": embedding_mode,
        "homolog_sequences": list(homolog_sequences),
        "query_sequence": query_sequence,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# ==========================================
# PHASE 1: HOMOLOG SEARCH (IN-MEMORY)
# ==========================================
def extract_homologs():
    print(f"\n--- Phase 1: Homolog Selection ({DEVICE}) ---")
    
    if USE_ONLY_EMBEDDINGS:
        print("Using ONLY embedding similarity for homolog selection (ignoring network file).")
        model_name_for_sim = SIMILARITY_MODEL_NAME if SIMILARITY_MODEL_NAME is not None else NETWORK_MODEL_NAME
        INPUT_EMBED_H5 = os.path.join(os.path.dirname(OUTPUT_HDF5), f"{SEQUENCE_SET}_[{model_name_for_sim}]_embeddings.h5")
        if not os.path.exists(INPUT_EMBED_H5):
            sys.exit(f"❌ Error: Embedding-only mode active, but base embedding file not found at: {INPUT_EMBED_H5}\n"
                     f"Please make sure you have generated the embeddings for {model_name_for_sim} first.")
            
        print(f"Loading base embeddings from {INPUT_EMBED_H5}...")
        mean_embs = []
        actual_seq_lens = []
        with h5py.File(INPUT_EMBED_H5, "r") as f_emb:
            raw_headers = f_emb['headers'][:]
            headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
            
            has_emb_group = "embeddings" in f_emb
            filtered_headers = []
            for h in tqdm(headers, desc="Loading Embeddings"):
                safe_h = h.replace("/", "_").replace("\\", "_")
                if has_emb_group:
                    emb = f_emb["embeddings"][safe_h][:]
                else:
                    emb = f_emb[safe_h][:]
                
                seq_len = emb.shape[0]
                if seq_len <= MAX_SEQ_LENGTH:
                    actual_seq_lens.append(seq_len)
                    filtered_headers.append(h)
                    if POOLING_METHOD == "max":
                        pooled = np.max(emb, axis=0)
                    else:
                        pooled = np.mean(emb, axis=0)
                    mean_embs.append(pooled)
                    
        num_valid = len(filtered_headers)
        print(f"Loaded {num_valid} valid sequences (<= {MAX_SEQ_LENGTH} AA) from embeddings.")
        if num_valid == 0:
            sys.exit("❌ Error: No valid sequences remaining after length filtering.")
            
        mean_embs = np.array(mean_embs, dtype=np.float32)
        actual_seq_lens = np.array(actual_seq_lens, dtype=np.float32)
        
        print("Calculating all-vs-all normalized embedding cosine similarities...")
        norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        norm_embs = mean_embs / norms
        cos_sim_mat = np.dot(norm_embs, norm_embs.T)
        cos_sim_mat = np.clip(cos_sim_mat, -1.0, 1.0)
        
        # Apply sequence length ratio adjustment if configured
        if LENGTH_RATIO_POWER != 0.0:
            lens_col = actual_seq_lens[:, np.newaxis]
            lens_row = actual_seq_lens[np.newaxis, :]
            min_lens = np.minimum(lens_col, lens_row)
            max_lens = np.maximum(lens_col, lens_row)
            max_lens = np.maximum(max_lens, 1)
            length_ratio_mat = min_lens / max_lens
            if LENGTH_RATIO_POWER != 1.0:
                length_ratio_mat = length_ratio_mat ** LENGTH_RATIO_POWER
            cos_sim_mat = cos_sim_mat * length_ratio_mat
            
        similarity_matrix = torch.tensor(cos_sim_mat, device=DEVICE, dtype=torch.float32)
        similarity_matrix.fill_diagonal_(float('-inf'))
        
    else:
        # Original network-based logic
        if not os.path.exists(INPUT_NETWORK_H5):
            sys.exit(f"❌ Error: Could not find network file: {INPUT_NETWORK_H5}")

        # ---> NEW: Detect if the input is a BLAST E-Value network <---
        is_blast = "EValue" in os.path.basename(INPUT_NETWORK_H5) or "blast" in os.path.basename(INPUT_NETWORK_H5).lower()
        if is_blast:
            print("Detected BLAST/E-Value Network. Bypassing embedding normalization.")

        print(f"Loading {INPUT_NETWORK_H5}...")
        with h5py.File(INPUT_NETWORK_H5, "r") as hf:
            raw_headers = hf['headers'][:]
            headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
            
            if is_blast:
                # BLAST networks do not store sequence lengths. Create a silent dummy array to bypass length filtering.
                seq_lens = np.zeros(len(headers), dtype=np.int32)
            elif 'seq_lens' in hf:
                seq_lens = hf['seq_lens'][:]
            else:
                print("Warning: 'seq_lens' not found. Creating dummy array.")
                seq_lens = np.zeros(len(headers), dtype=np.int32)

            num_original = len(headers)
            
            src_indices = hf['i'][:].astype(np.int64)
            dst_indices = hf['j'][:].astype(np.int64)
            
            # ---> NEW: Safely extract scores based on network type <---
            if is_blast:
                raw_scores = hf['score'][:].astype(np.float32)
                raw_lens = None
            else:
                if SCORE_MODE == "global":
                    raw_scores = hf['g_score'][:].astype(np.float32)
                    raw_lens   = hf['g_len'][:].astype(np.float32)
                else:
                    raw_scores = hf['l_score'][:].astype(np.float32)
                    raw_lens   = hf['l_len'][:].astype(np.float32)

        # Filter Sequences > MAX_SEQ_LENGTH
        valid_mask = seq_lens <= MAX_SEQ_LENGTH
        valid_indices = np.where(valid_mask)[0]
        num_valid = len(valid_indices)
        
        print(f"Original sequences: {num_original} | Valid (<= {MAX_SEQ_LENGTH} AA): {num_valid}")
        if num_valid == 0:
            sys.exit("❌ Error: No valid sequences remaining after length filtering.")

        old_to_new_map = np.full(num_original, -1, dtype=np.int64)
        old_to_new_map[valid_indices] = np.arange(num_valid)

        # Build Dense Similarity Matrix
        print("Constructing Similarity Matrix...")
        
        valid_edge_mask = (valid_mask[src_indices]) & (valid_mask[dst_indices])
        final_src = old_to_new_map[src_indices[valid_edge_mask]]
        final_dst = old_to_new_map[dst_indices[valid_edge_mask]]
        final_scores = torch.tensor(raw_scores[valid_edge_mask], device=DEVICE, dtype=torch.float32)

        t_src = torch.tensor(final_src, device=DEVICE, dtype=torch.long)
        t_dst = torch.tensor(final_dst, device=DEVICE, dtype=torch.long)
        
        # ---> NEW: Bypass Normalization entirely for BLAST E-Values <---
        if is_blast:
            normalized_scores = final_scores
        else:
            if NORMALIZATION_MODE == "alignment_length":
                align_lens = torch.tensor(raw_lens[valid_edge_mask], device=DEVICE, dtype=torch.float32)
                denom = align_lens + 1e-6
            else:
                valid_lens_gpu = torch.tensor(seq_lens[valid_indices], device=DEVICE, dtype=torch.float32)
                len_src, len_dst = valid_lens_gpu[t_src], valid_lens_gpu[t_dst]
                
                if NORMALIZATION_MODE == "longer_sequence": denom = torch.maximum(len_src, len_dst) + 1e-6
                elif NORMALIZATION_MODE == "shorter_sequence": denom = torch.minimum(len_src, len_dst) + 1e-6
                elif NORMALIZATION_MODE == "average_sequence": denom = ((len_src + len_dst) / 2.0) + 1e-6
                else: sys.exit(f"❌ Error: Unknown normalization mode '{NORMALIZATION_MODE}'")

            normalized_scores = final_scores / denom

        is_sparse = len(final_src) < int(num_valid * (num_valid - 1) / 2)
        
        if is_sparse and not is_blast:
            print(f"Network is sparse ({len(final_src)} / {int(num_valid * (num_valid - 1) / 2)} edges). Activating hybrid cosine-alignment transformation...")
            model_name_for_sim = SIMILARITY_MODEL_NAME if SIMILARITY_MODEL_NAME is not None else NETWORK_MODEL_NAME
            INPUT_EMBED_H5 = os.path.join(os.path.dirname(OUTPUT_HDF5), f"{SEQUENCE_SET}_[{model_name_for_sim}]_embeddings.h5")
            if not os.path.exists(INPUT_EMBED_H5):
                sys.exit(f"❌ Error: Sparse network detected, but base embedding file not found at: {INPUT_EMBED_H5}\n"
                         f"Please make sure you have generated the embeddings for {model_name_for_sim} first.")
                
            print(f"Loading base embeddings from {INPUT_EMBED_H5}...")
            mean_embs = []
            actual_seq_lens = []
            filtered_headers = [headers[i] for i in valid_indices]
            with h5py.File(INPUT_EMBED_H5, "r") as f_emb:
                has_emb_group = "embeddings" in f_emb
                for h in tqdm(filtered_headers, desc="Computing Pooled Embeddings"):
                    safe_h = h.replace("/", "_").replace("\\", "_")
                    if has_emb_group:
                        emb = f_emb["embeddings"][safe_h][:]
                    else:
                        emb = f_emb[safe_h][:]
                    
                    actual_seq_lens.append(emb.shape[0])
                    if POOLING_METHOD == "max":
                        pooled = np.max(emb, axis=0)
                    else:
                        pooled = np.mean(emb, axis=0)
                    mean_embs.append(pooled)
                    
            mean_embs = np.array(mean_embs, dtype=np.float32)
            actual_seq_lens = np.array(actual_seq_lens, dtype=np.float32)
            
            print("Calculating all-vs-all length-adjusted similarities (length ratio * cosine similarity)...")
            norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            norm_embs = mean_embs / norms
            cos_sim_mat = np.dot(norm_embs, norm_embs.T)
            cos_sim_mat = np.clip(cos_sim_mat, -1.0, 1.0)
            
            # Apply sequence length ratio adjustment
            lens_col = actual_seq_lens[:, np.newaxis]
            lens_row = actual_seq_lens[np.newaxis, :]
            min_lens = np.minimum(lens_col, lens_row)
            max_lens = np.maximum(lens_col, lens_row)
            max_lens = np.maximum(max_lens, 1)
            length_ratio_mat = min_lens / max_lens
            
            if LENGTH_RATIO_POWER != 1.0:
                length_ratio_mat = length_ratio_mat ** LENGTH_RATIO_POWER
                
            cos_sim_mat = cos_sim_mat * length_ratio_mat
            
            # Extract overlapping pairs
            X_cos = cos_sim_mat[final_src, final_dst]
            Y_align = normalized_scores.cpu().numpy()
            
            print("Fitting Isotonic Regression (Adjusted Similarity -> Alignment Score)...")
            if len(X_cos) > 100000:
                np.random.seed(42)
                sample_idx = np.random.choice(len(X_cos), size=100000, replace=False)
                X_fit = X_cos[sample_idx]
                Y_fit = Y_align[sample_idx]
            else:
                X_fit = X_cos
                Y_fit = Y_align
                
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(X_fit, Y_fit)
            
            # Evaluate fitness on all edges
            Y_pred = iso_reg.predict(X_cos)
            rho, _ = spearmanr(X_cos, Y_align)
            r2 = r2_score(Y_align, Y_pred)
            print(f"Isotonic Regression Fit Diagnostics:")
            print(f"  > Spearman Rank Correlation (rho): {rho:.4f}")
            print(f"  > Coefficient of Determination (R^2): {r2:.4f}")
            
            print("Imputing missing similarity scores for all-vs-all pairs...")
            dense_scores = iso_reg.predict(cos_sim_mat.ravel()).reshape(num_valid, num_valid)
            dense_scores = 0.5 * (dense_scores + dense_scores.T)
            
            similarity_matrix = torch.tensor(dense_scores, device=DEVICE, dtype=torch.float32)
        else:
            similarity_matrix = torch.full((num_valid, num_valid), float('-inf'), device=DEVICE, dtype=torch.float32)
            
        similarity_matrix[t_src, t_dst] = normalized_scores
        similarity_matrix[t_dst, t_src] = normalized_scores 
        similarity_matrix.fill_diagonal_(float('-inf'))
        
        del src_indices, dst_indices, raw_scores, raw_lens, final_src, final_dst, final_scores, t_src, t_dst
        torch.cuda.empty_cache()
        filtered_headers = [headers[i] for i in valid_indices]

    # Select Homologs
    print(f"Selecting Homologs (Stride {SELECTION_STRIDE})...")
    actual_k = min(TOP_K_SEARCH, num_valid - 1) 
    target_indices = torch.arange(SELECTION_STRIDE - 1, TOP_K_SEARCH, SELECTION_STRIDE, device=DEVICE)
    
    if actual_k < SELECTION_STRIDE:
        target_indices = torch.tensor([0], device=DEVICE); actual_k = 1

    _, top_indices = torch.topk(similarity_matrix, k=actual_k, dim=1)
    
    valid_target_mask = target_indices < actual_k
    valid_target_indices = target_indices[valid_target_mask]
    selected_homologs = top_indices[:, valid_target_indices].cpu().numpy()
    
    # Create Dictionary mapping Header -> List of Homolog Headers
    homolog_map = {}
    for i in range(num_valid):
        query_header = filtered_headers[i]
        homolog_headers = [filtered_headers[h_idx] for h_idx in selected_homologs[i]]
        homolog_map[query_header] = homolog_headers
        
    print("✅ Homolog extraction complete (Kept in memory).")
    return filtered_headers, homolog_map

# ==========================================
# PHASE 2: EMBEDDING GENERATION
# ==========================================
def load_e1_model(model_name, device):
    print(f"\nLoading {model_name} on {device}...")
    model = E1ForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    batch_preparer = E1BatchPreparer()
    return model, batch_preparer

def get_e1_embeddings_batch(input_strings, model, batch_preparer, device, target_dtype):
    batch = batch_preparer.get_batch_kwargs(input_strings, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            outputs = model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                use_cache=False, output_attentions=False, output_hidden_states=False,
            )

    embeddings = outputs.embeddings 
    last_seq_ids = batch["sequence_ids"].max(dim=1)[0][:, None]
    last_sequence_mask = batch["sequence_ids"] == last_seq_ids
    valid_residue_mask = ~batch_preparer.get_boundary_token_mask(batch["input_ids"])
    query_mask = last_sequence_mask & valid_residue_mask

    batch_results = []
    for i in range(len(input_strings)):
        emb_i = embeddings[i, query_mask[i]]
        batch_results.append(emb_i.cpu().float().numpy().astype(target_dtype))

    return batch_results


def prepare_e1_inputs(
    fasta_headers,
    fasta_sequences,
    embedding_mode,
    checkpoint,
    *,
    target_headers=None,
    homolog_map=None,
    max_homologs=MAX_HOMOLOGS,
):
    """Resolve the exact query/context strings and hashes for E1 generation."""
    validate_manifest_records(fasta_headers, fasta_sequences)
    sequence_by_header = dict(zip(fasta_headers, fasta_sequences))

    if embedding_mode == "NORM":
        resolved_targets = list(fasta_headers)
        homolog_map = {}
    elif embedding_mode == "RA":
        if target_headers is None or homolog_map is None:
            raise ValueError(
                "RA mode requires both target_headers and a homolog_map."
            )
        resolved_targets = list(target_headers)
    else:
        raise ValueError("embedding_mode must be exactly 'RA' or 'NORM'.")

    if len(resolved_targets) != len(set(resolved_targets)):
        raise ValueError("E1 target headers contain duplicates.")

    missing_targets = [
        header for header in resolved_targets if header not in sequence_by_header
    ]
    if missing_targets:
        raise ValueError(
            f"E1 target header '{missing_targets[0]}' is absent from the "
            "sanitized FASTA records. Regenerate the network from the same "
            "sanitized sequence set."
        )

    query_sequences = []
    input_strings = []
    input_hashes = []

    for header in resolved_targets:
        query_sequence = sequence_by_header[header]
        homolog_sequences = []

        if embedding_mode == "RA":
            if header not in homolog_map:
                raise ValueError(
                    f"RA homolog map is missing target header '{header}'."
                )
            homolog_headers = list(homolog_map[header])[:max_homologs]
            missing_homologs = [
                homolog_header
                for homolog_header in homolog_headers
                if homolog_header not in sequence_by_header
            ]
            if missing_homologs:
                raise ValueError(
                    f"RA homolog header '{missing_homologs[0]}' for target "
                    f"'{header}' is absent from the sanitized FASTA records. "
                    "Regenerate the network from the same sanitized sequence set."
                )
            homolog_sequences = [
                sequence_by_header[homolog_header]
                for homolog_header in homolog_headers
            ]

        combined_sequences = homolog_sequences + [query_sequence]
        input_string = ",".join(combined_sequences)
        input_hash = build_input_hash(
            checkpoint,
            embedding_mode,
            homolog_sequences,
            query_sequence,
        )

        query_sequences.append(query_sequence)
        input_strings.append(input_string)
        input_hashes.append(input_hash)

    validate_manifest_records(resolved_targets, query_sequences)
    validate_input_hashes(input_hashes, len(resolved_targets))
    return resolved_targets, query_sequences, input_strings, input_hashes


def generate_embeddings(
    input_fasta,
    output_hdf5,
    embedding_mode,
    saving_mode,
    *,
    batch_size=BATCH_SIZE,
    max_homologs=MAX_HOMOLOGS,
    homolog_selector=extract_homologs,
    model_loader=load_e1_model,
    embedding_batcher=get_e1_embeddings_batch,
    device=DEVICE,
):
    """Generate or safely resume one standalone E1 embedding database."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if (
        not isinstance(max_homologs, int)
        or isinstance(max_homologs, bool)
        or max_homologs < 0
    ):
        raise ValueError("max_homologs must be a non-negative integer.")

    normalized_mode, model_name, checkpoint = resolve_e1_model(embedding_mode)
    target_dtype = dtype_for_saving_mode(saving_mode)
    fasta_headers, fasta_sequences, _ = load_sanitized_fasta(input_fasta)
    validate_manifest_records(fasta_headers, fasta_sequences)
    print(f"Loaded {len(fasta_headers)} sanitized sequences from FASTA.")

    target_headers = None
    homolog_map = None
    if normalized_mode == "RA":
        target_headers, homolog_map = homolog_selector()

    (
        output_headers,
        output_sequences,
        input_strings,
        input_hashes,
    ) = prepare_e1_inputs(
        fasta_headers,
        fasta_sequences,
        normalized_mode,
        checkpoint,
        target_headers=target_headers,
        homolog_map=homolog_map,
        max_homologs=max_homologs,
    )

    output_directory = os.path.dirname(os.fspath(output_hdf5))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    print(f"Opening {output_hdf5} for evaluation...")

    with h5py.File(output_hdf5, "a") as hf:
        is_new_file = len(hf) == 0 and len(hf.attrs) == 0
        if is_new_file:
            emb_group = create_metadata_first_file(
                hf,
                output_headers,
                output_sequences,
                input_hashes,
                model_name,
                saving_mode,
            )
            feature_dimension = None
            stale_headers = []
        else:
            existing = read_embedding_manifest(
                hf,
                require_complete=False,
                validate_embeddings=True,
            )
            if existing.model_name != model_name:
                raise ValueError(
                    f"Cannot resume model '{model_name}' from a file created "
                    f"with model '{existing.model_name}'."
                )
            if existing.saving_mode != saving_mode:
                raise ValueError(
                    f"Cannot resume {saving_mode} generation from a "
                    f"{existing.saving_mode} file."
                )

            incoming_by_header = dict(zip(output_headers, output_sequences))
            existing_by_header = existing.sequence_by_header
            removed = sorted(set(existing.headers) - set(output_headers))
            if removed:
                raise ValueError(
                    f"Cannot resume because existing header '{removed[0]}' "
                    "was removed from the sanitized target manifest."
                )
            changed = sorted(
                header
                for header in set(existing.headers) & set(output_headers)
                if existing_by_header[header] != incoming_by_header[header]
            )
            if changed:
                raise ValueError(
                    "Cannot resume because the sanitized query sequence for "
                    f"'{changed[0]}' changed."
                )

            incoming_hash_by_header = dict(zip(output_headers, input_hashes))
            existing_hash_by_header = existing.input_hash_by_header
            stale_headers = [
                header
                for header in output_headers
                if header in existing_hash_by_header
                and existing_hash_by_header[header]
                != incoming_hash_by_header[header]
            ]

            if stale_headers:
                hf.attrs["generation_complete"] = False
                hf.flush()
                for header in stale_headers:
                    if header in hf["embeddings"]:
                        del hf["embeddings"][header]
                hf.flush()
                print(
                    "Regenerating context-changed embeddings: "
                    f"{len(stale_headers)} stale record(s)."
                )

            manifest_changed = (
                existing.headers != output_headers
                or existing.sequences != output_sequences
                or existing.input_hashes != input_hashes
            )
            if manifest_changed:
                write_embedding_manifest(
                    hf,
                    output_headers,
                    output_sequences,
                    input_hashes,
                    model_name,
                    saving_mode,
                    replace=True,
                )

            emb_group = hf["embeddings"]
            feature_dimension = existing.feature_dimension

        input_by_header = dict(zip(output_headers, input_strings))
        sequence_by_header = dict(zip(output_headers, output_sequences))
        existing_keys = set(emb_group.keys())
        pending_headers = [
            header for header in output_headers if header not in existing_keys
        ]

        if not pending_headers:
            if not bool(hf.attrs["generation_complete"]):
                mark_generation_complete(hf)
            print(
                f"HDF5 database already complete ({len(output_headers)} embeddings). "
                "Skipping generation."
            )
            return 0

        hf.attrs["generation_complete"] = False
        hf.flush()
        if existing_keys:
            print(
                f"Resuming from interruption: {len(existing_keys)} found, "
                f"{len(pending_headers)} remaining."
            )
        else:
            print(
                "Starting fresh E1 embedding generation for "
                f"{len(pending_headers)} sequences."
            )

        if model_loader is load_e1_model and not HAS_E1:
            raise ImportError(
                "Could not import 'E1'. Install the E1 repository with "
                "'pip install -e .' in the WSL environment."
            )
        model, batch_preparer = model_loader(checkpoint, device)

        def validate_and_store(batch_headers, batch_embeddings):
            nonlocal feature_dimension
            if len(batch_embeddings) != len(batch_headers):
                raise ValueError(
                    "E1 returned a different number of embeddings than inputs."
                )

            next_feature_dimension = feature_dimension
            for header, embedding in zip(batch_headers, batch_embeddings):
                next_feature_dimension = validate_embedding_array(
                    embedding,
                    sequence_by_header[header],
                    saving_mode,
                    feature_dimension=next_feature_dimension,
                    require_finite=True,
                    header=header,
                )

            for header, embedding in zip(batch_headers, batch_embeddings):
                emb_group.create_dataset(header, data=embedding)
                hf.flush()
            feature_dimension = next_feature_dimension

        for start in tqdm(
            range(0, len(pending_headers), batch_size),
            desc="Batches",
        ):
            batch_headers = pending_headers[start:start + batch_size]
            batch_inputs = [input_by_header[header] for header in batch_headers]

            try:
                batch_embeddings = embedding_batcher(
                    batch_inputs,
                    model,
                    batch_preparer,
                    device,
                    target_dtype,
                )
                validate_and_store(batch_headers, batch_embeddings)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise

                print(
                    f"\nOOM on batch {start // batch_size}. "
                    "Retrying its records sequentially..."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for header, input_string in zip(batch_headers, batch_inputs):
                    try:
                        single_embedding = embedding_batcher(
                            [input_string],
                            model,
                            batch_preparer,
                            device,
                            target_dtype,
                        )
                        validate_and_store([header], single_embedding)
                    except RuntimeError as single_exc:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(
                            f"E1 generation failed for '{header}' after "
                            "single-sequence OOM recovery. The output remains "
                            "incomplete and can be resumed."
                        ) from single_exc

        mark_generation_complete(hf)
        return len(pending_headers)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"--- Integrated E1 Embedding Generation [{EMBEDDING_MODE} Mode] ---")
    print(f"--> Precision: {SAVING_MODE}")
    try:
        generated_count = generate_embeddings(
            INPUT_FASTA,
            OUTPUT_HDF5,
            EMBEDDING_MODE,
            SAVING_MODE,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if generated_count:
        print(
            f"Done! {generated_count} embeddings generated and saved to "
            f"{OUTPUT_HDF5}."
        )
