#!/usr/bin/env bash
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

# EMAP-SSN-E1-Embedding-Generator: set up the project's Python environment and work in it.
#
#   ./start.sh                  check .venv, build or repair it when needed, then open a shell in it
#   ./start.sh <command> [...]  the same, but run the command in the environment instead, e.g.
#                               ./start.sh python Generate_Embeddings_E1_fast.py
#   ./start.sh --check-only     only report: exit 0 when .venv is ready, 10 when setup is needed
#                               (add --import to also import torch, flash-attn and E1; uses the GPU)
#   ./start.sh --setup-only     build or repair .venv, then exit
#
# Needs Linux (or WSL2) on x86_64 with an NVIDIA GPU and driver. The environment is .venv in this
# folder, made with uv (https://docs.astral.sh/uv/); uv is downloaded to ~/.local/bin when it is
# missing, and nothing else outside this folder is changed. Package versions are pinned to those
# the published embeddings were checked with (setup/requirements.txt, setup/constraints.txt), and
# Profluent-E1 is installed from third_party/E1 (editable).

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Run ./start.sh (or bash start.sh); do not source it." >&2
    return 1
fi
set -u

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 1
cd "$PROJECT_ROOT" || exit 1
VENV=.venv
VENV_PYTHON=$VENV/bin/python
PYTHON_VERSION=3.12

# The environment's interpreter must see its own packages only (conda and user settings
# can point Python elsewhere).
unset PYTHONHOME PYTHONPATH PYTHONSAFEPATH

# Prints how to run uv, or fails when it is not installed: PATH first, then ~/.local/bin,
# where uv's own installer puts it.
e1_find_uv() {
    if command -v uv >/dev/null 2>&1; then
        printf '%s\n' uv
        return 0
    fi
    if [ -x "$HOME/.local/bin/uv" ]; then
        printf '%s\n' "$HOME/.local/bin/uv"
        return 0
    fi
    return 1
}

# Downloads uv into ~/.local/bin and sets UV_EXE. Only setup calls this, never the check.
# UV_NO_MODIFY_PATH=1 keeps the installer away from the shell start-up files.
e1_install_uv() {
    echo "uv package manager not found. Installing it into ~/.local/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "Installing uv failed. Install uv from https://docs.astral.sh/uv/ and run this again." >&2
        return 1
    fi
    UV_EXE="$HOME/.local/bin/uv"
    echo "uv is installed in ~/.local/bin. Your shell start-up files were not changed."
}

e1_venv_python_ok() {
    [ -x "$VENV_PYTHON" ] &&
        "$VENV_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1
}

# A CUDA compiler lets flash-attn fall back to compiling itself when no prebuilt wheel fits.
e1_have_nvcc() {
    command -v nvcc >/dev/null 2>&1 ||
        [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ] || [ -x /usr/local/cuda/bin/nvcc ]
}

# Read-only. Exit status: 0 ready, 10 setup needed. Extra arguments go to setup/check_env.py.
e1_check() {
    if ! e1_venv_python_ok; then
        echo "Setup is needed: $VENV is missing or does not run Python $PYTHON_VERSION." >&2
        return 10
    fi
    "$VENV_PYTHON" setup/check_env.py "$@"
}

e1_setup() {
    local uv
    if ! uv=$(e1_find_uv); then
        e1_install_uv || return 1
        uv=$UV_EXE
    fi
    if ! e1_venv_python_ok; then
        echo "Creating the local Python environment ($VENV)..."
        # uv's own Python first, so that updating or removing another Python (conda, the
        # system's) cannot break the environment.
        if ! UV_PYTHON_PREFERENCE=only-managed "$uv" venv --clear --python "$PYTHON_VERSION" --prompt e1 "$VENV"; then
            echo "uv's own Python $PYTHON_VERSION is not available; using a Python $PYTHON_VERSION found on this computer."
            "$uv" venv --clear --python "$PYTHON_VERSION" --prompt e1 "$VENV" || return 1
        fi
    fi
    echo "Installing the pinned packages (about 4 GB to download the first time)..."
    "$uv" pip install --python "$VENV_PYTHON" -r setup/requirements.txt -c setup/constraints.txt || return 1

    echo "Installing flash-attn (its setup downloads the prebuilt wheel that matches torch)..."
    if ! e1_have_nvcc; then
        # No CUDA compiler: take the prebuilt wheel or fail, never a build without the GPU code.
        export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
    else
        echo "  (If no prebuilt wheel fits, it compiles with nvcc instead: that can take hours.)"
    fi
    MAX_JOBS=${MAX_JOBS:-4} "$uv" pip install --python "$VENV_PYTHON" --no-build-isolation \
        -r setup/flash_attn.txt -c setup/constraints.txt || return 1

    echo "Installing Profluent-E1 from third_party/E1 (editable)..."
    "$uv" pip install --python "$VENV_PYTHON" --no-deps -e third_party/E1 || return 1
}

# Runs the command in the environment, or opens an interactive shell in it.
e1_enter() {
    local rc="$VENV/e1_shell_rc"

    if [ "$#" -gt 0 ]; then
        # shellcheck disable=SC1091
        . "$VENV/bin/activate"
        exec "$@"
    fi
    cat > "$rc" <<'EOF'
# Written by start.sh: start-up file of the shell it opens. Your own ~/.bashrc runs first,
# then this project's .venv is activated on top of whatever that set up (conda included).
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
unset PYTHONHOME PYTHONPATH PYTHONSAFEPATH
. "$E1_PROJECT_ROOT/.venv/bin/activate"
cd "$E1_PROJECT_ROOT" || true
echo "E1 environment: $(python -V) from $E1_PROJECT_ROOT/.venv. Type exit to leave."
EOF
    export E1_PROJECT_ROOT="$PROJECT_ROOT"
    exec bash --rcfile "$rc" -i
}

case "${1:-}" in
    -h|--help)
        # The usage block: from its title line to the first blank line after it
        sed -n '/^# EMAP-SSN-E1-Embedding-Generator:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    --check-only)
        shift
        e1_check "$@"
        exit $?
        ;;
    --setup-only)
        e1_setup || { echo "Setup failed." >&2; exit 1; }
        e1_check
        exit $?
        ;;
esac

e1_check
status=$?
if [ "$status" -ne 0 ]; then
    if [ "$status" -ne 10 ]; then
        echo "The environment check failed (exit $status)." >&2
        exit 1
    fi
    e1_setup || { echo "Setup failed." >&2; exit 1; }
    e1_check || { echo "The environment is still not ready; see the messages above." >&2; exit 1; }
fi
e1_enter "$@"
