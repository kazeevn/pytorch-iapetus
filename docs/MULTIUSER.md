# Multi-User Docker Setup for PyTorch & CUDA Workstations

## 1. Problem Statement

When sharing Docker images across multiple local users (such as `kna` with UID 1000 and `maevskiy` with UID 1001) on a shared GPU workstation, common problems arise:

1. **Hardcoded User in Image (`USER kna`)**:
   - If user `maevskiy` runs the image, they run as user `kna` inside the container.
   - Any files written to a mounted directory are owned by UID 1000 (`kna`), causing permission conflicts when `maevskiy` accesses them on the host.
   - If `maevskiy` mounts their private directory (`/home/maevskiy/...`), user `kna` gets `Permission Denied` on the host files.
2. **Defaulting to `root` (`USER root`)**:
   - Files created inside the container are owned by `root:root`, requiring `sudo` on the host to edit or remove them.
3. **Simple `--user $(id -u):$(id -g)` without User Setup**:
   - Arbitrary UIDs have no entry in `/etc/passwd` (`whoami` fails with "cannot find name for user ID").
   - `$HOME` defaults to `/` or `/home/kna`, causing permission denied errors on cache directories (`~/.cache`, PyTorch hub, pip).
   - The user is not in the `video` and `render` groups, which can cause CUDA device permission issues.

---

## 2. Architecture: Dynamic Entrypoint Pattern

The cleanest solution is the **Dynamic Entrypoint Pattern**:
* The container engine starts the container as `root`.
* An `entrypoint.sh` script runs before any user command.
* The script determines the invoking user's UID and GID (either auto-detected from the mounted `/workspace` directory or passed via environment variables).
* It dynamically creates the user account in `/etc/passwd` and `/etc/group` (if not already present), adds them to `video` and `render` groups for GPU access, and configures `$HOME`.
* It drops privileges using Linux's standard `setpriv` command and executes the command as that user.

```
┌──────────────────────────────────────────────────────────┐
│ Container starts as root (UID 0)                         │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│ entrypoint.sh:                                           │
│ 1. Read TARGET_UID/GID from /workspace or ENV            │
│ 2. Create user & group if missing                        │
│ 3. Add to 'video' and 'render' groups (GPU access)       │
│ 4. Configure $HOME and $PATH                             │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│ exec setpriv --reuid=$TARGET_UID --regid=$TARGET_GID ... │
│ Command runs as the invoking host user                   │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Implementation

### 3.1. `entrypoint.sh`

The entrypoint lives in `docker/entrypoint.sh`:

```bash
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
fi

export HOME="$TARGET_HOME"
export USER="$TARGET_USER"
export PATH="/opt/venv312/bin:$PATH"

# 6. Drop privileges and execute command
exec setpriv --reuid="$TARGET_UID" --regid="$TARGET_GID" --init-groups "$@"
```

### 3.2. Runtime image (`docker/runtime.Dockerfile`)

The entrypoint is installed by `docker/runtime.Dockerfile`. The original standalone wrapper looked like this (kept for reference; see `archive/legacy-docker/Dockerfile.universal`):

```dockerfile
FROM pytorch:2.14.0-cuda11.8-py312

USER root

# Install entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Ensure container starts as root so entrypoint can configure users
USER root
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
```

### 3.3. Build Command

```bash
scripts/build_images.sh runtime   # tags iapetus/pytorch:2.14.0-cuda11.8-py312
```

---

## 4. Usage Instructions

### Zero-Config Auto-Detection (Recommended)

When a user bind-mounts their directory to `/workspace`, the entrypoint reads the directory ownership and automatically runs as that user:

#### For User `kna` (UID 1000):
```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  iapetus/pytorch:2.14.0-cuda11.8-py312
```
* Container user: `kna` (UID 1000)
* All created files belong to `kna:kna`.

#### For User `maevskiy` (UID 1001):
```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v /home/maevskiy/pytorch-research:/workspace \
  iapetus/pytorch:2.14.0-cuda11.8-py312
```
* Container user: `maevskiy` (UID 1001)
* All created files belong to `maevskiy:maevskiy`.

---

### Explicit User Pass (Optional)

Users can explicitly specify their host identity regardless of mount location:

```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e HOST_USER=$(whoami) \
  --ipc=host \
  -v "$(pwd):/workspace" \
  iapetus/pytorch:2.14.0-cuda11.8-py312
```

---

### Running as Root (Administrative Tasks)

To run as `root` (for installing system packages inside a test container):

```bash
docker run --rm -it \
  -e HOST_UID=0 \
  iapetus/pytorch:2.14.0-cuda11.8-py312 \
  bash
```

---

## 5. Key Advantages & Notes

* **No Extra Dependencies**: Uses `setpriv`, which is built into standard Linux (`util-linux`), avoiding external binaries like `gosu` or `su-exec`.
* **CUDA Hardware Permissions**: Automatically adds the detected user to `video` and `render` groups so `/dev/nvidia*` and `/dev/dri/*` devices are accessible.
* **Persistent Home Caches**: Each user gets their own `/home/<username>` inside the container with proper ownership, preventing `Permission denied` on `~/.cache` and PyTorch model downloads.
* **Host Portability**: Works for any new user added to this machine without rebuilding the Docker image.
