#!/bin/bash
set -e

# 1. Determine Target UID and GID:
#    Priority 1: Explicit environment variables HOST_UID / HOST_GID if supplied.
#    Priority 2: Auto-detect owner of /workspace if mounted from host.
#    Priority 3: Fallback to UID/GID 1000.
TARGET_UID="${HOST_UID:-$(stat -c '%u' /workspace 2>/dev/null || echo 1000)}"
TARGET_GID="${HOST_GID:-$(stat -c '%g' /workspace 2>/dev/null || echo 1000)}"
TARGET_USER="${HOST_USER:-developer}"

# If root (UID 0) was explicitly requested, bypass privilege drop
if [ "$TARGET_UID" -eq 0 ]; then
    exec "$@"
fi

# 2. Dynamically create group if missing
if ! getent group "$TARGET_GID" >/dev/null 2>&1; then
    groupadd -g "$TARGET_GID" "$TARGET_USER" 2>/dev/null || true
fi

TARGET_GROUP=$(getent group "$TARGET_GID" | cut -d: -f1)

# 3. Dynamically create user if missing
if ! id -u "$TARGET_UID" >/dev/null 2>&1; then
    useradd -u "$TARGET_UID" -g "$TARGET_GID" -m -s /bin/bash "$TARGET_USER" 2>/dev/null || true
    TARGET_HOME="/home/$TARGET_USER"
else
    TARGET_USER=$(id -un "$TARGET_UID")
    TARGET_HOME=$(getent passwd "$TARGET_UID" | cut -d: -f6)
fi

# 4. Add user to hardware groups needed for NVIDIA GPU access
usermod -aG video,render "$TARGET_USER" 2>/dev/null || true

# 5. Ensure Python venv is in PATH and user's .bashrc
if [ -d "$TARGET_HOME" ]; then
    if ! grep -q "/opt/venv312/bin" "$TARGET_HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="/opt/venv312/bin:$PATH"' >> "$TARGET_HOME/.bashrc"
    fi
    if ! grep -q "oneapi/setvars.sh" "$TARGET_HOME/.bashrc" 2>/dev/null; then
        echo 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true' >> "$TARGET_HOME/.bashrc"
    fi
    if ! grep -q "USE_LIBUV" "$TARGET_HOME/.bashrc" 2>/dev/null; then
        echo 'export USE_LIBUV=0' >> "$TARGET_HOME/.bashrc"
    fi
    if ! grep -q "CMAKE_PREFIX_PATH" "$TARGET_HOME/.bashrc" 2>/dev/null; then
        echo 'export CMAKE_PREFIX_PATH="/usr/local:/opt/venv312/lib/python3.12/site-packages/torch/share/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor/lib/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor_torch/torch-2.14/lib/cmake:${CMAKE_PREFIX_PATH:-}"' >> "$TARGET_HOME/.bashrc"
    fi
fi

export HOME="$TARGET_HOME"
export USER="$TARGET_USER"
export PATH="/opt/venv312/bin:$PATH"
export USE_LIBUV=0
export CMAKE_PREFIX_PATH="/usr/local:/opt/venv312/lib/python3.12/site-packages/torch/share/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor/lib/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor_torch/torch-2.14/lib/cmake:${CMAKE_PREFIX_PATH:-}"

# 6. Drop privileges and execute command
exec setpriv --reuid="$TARGET_UID" --regid="$TARGET_GID" --init-groups "$@"

