# Multi-User Docker Setup for PyTorch & CUDA Workstations

## 1. Problem Statement

When sharing Docker images across multiple local users (such as `alice` with UID 1000 and `bob` with UID 1001) on a shared GPU workstation, common problems arise:

1. **Hardcoded User in Image (`USER alice`)**:
   - If user `bob` runs the image, they run as user `alice` inside the container.
   - Any files written to a mounted directory are owned by UID 1000 (`alice`), causing permission conflicts when `bob` accesses them on the host.
   - If `bob` mounts their private directory (`/home/bob/...`), user `alice` gets `Permission Denied` on the host files.
2. **Defaulting to `root` (`USER root`)**:
   - Files created inside the container are owned by `root:root`, requiring `sudo` on the host to edit or remove them.
3. **Simple `--user $(id -u):$(id -g)` without User Setup**:
   - Arbitrary UIDs have no entry in `/etc/passwd` (`whoami` fails with "cannot find name for user ID").
   - `$HOME` defaults to `/` or `/home/alice`, causing permission denied errors on cache directories (`~/.cache`, PyTorch hub, pip).
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

[`docker/entrypoint.sh`](../docker/entrypoint.sh) runs as root at container start and:

1. Picks the target UID/GID: `HOST_UID`/`HOST_GID` if set, otherwise the owner of `/workspace`, otherwise 1000.
   If UID 0 is requested, it runs the command as root without further setup.
2. Creates the group and user (`HOST_USER`, or a generated name) if they don't exist, with a home directory
   under `/home`.
3. Adds the user to the `video` and `render` groups for GPU device access.
4. Writes `PATH` (`/opt/venv312/bin`), the oneAPI environment, `USE_LIBUV=0` and `CMAKE_PREFIX_PATH` into the
   user's `.bashrc` and exports them for the command.
5. Drops privileges with `setpriv --reuid --regid --init-groups` and `exec`s the command.

### 3.2. Runtime image (`docker/runtime.Dockerfile`)

The runtime image installs the entrypoint and starts as root so that the entrypoint can create the user:

```dockerfile
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
```

### 3.3. Build Command

```bash
scripts/build_images.sh runtime   # tags iapetus/pytorch:2.14.0-cuda11.8-cudnn8.7-py312
```

---

## 4. Usage Instructions

### Zero-Config Auto-Detection (Recommended)

When a user bind-mounts their directory to `/workspace`, the entrypoint reads the directory's owner UID/GID and runs as that UID/GID. Inside the container the account is named `developer` unless `HOST_USER` is set (see below); on the host, files belong to the real user because the UID matches:

#### For User `alice` (UID 1000):
```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus
```
* Container user: `developer` (UID 1000)
* On the host, created files belong to `alice:alice`.

#### For User `bob` (UID 1001):
```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v /home/bob/pytorch-research:/workspace \
  ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus
```
* Container user: `developer` (UID 1001)
* On the host, created files belong to `bob:bob`.

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
  ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus
```

---

### Running as Root (Administrative Tasks)

To run as `root` (for installing system packages inside a test container):

```bash
docker run --rm -it \
  -e HOST_UID=0 \
  ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus \
  bash
```

---

## 5. Key Advantages & Notes

* **No Extra Dependencies**: Uses `setpriv`, which is built into standard Linux (`util-linux`), avoiding external binaries like `gosu` or `su-exec`.
* **CUDA Hardware Permissions**: Automatically adds the detected user to `video` and `render` groups so `/dev/nvidia*` and `/dev/dri/*` devices are accessible.
* **Per-User Home**: Each user gets their own `/home/<username>` inside the container with proper ownership, preventing `Permission denied` on `~/.cache` and PyTorch model downloads.
* **Host Portability**: Works for any new user added to this machine without rebuilding the Docker image.
