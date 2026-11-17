#!/usr/bin/env bash
set -euo pipefail

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

export HF_HOME="${HF_HOME:-/data/models}"
export AURISDECK_DATA_DIR="${AURISDECK_DATA_DIR:-/data}"

# Expose pip-installed NVIDIA cuBLAS/cuDNN shared libraries to CTranslate2/PyTorch
shopt -s nullglob
nvidia_lib_dirs=(/opt/aurisdeck/.venv/lib/python*/site-packages/nvidia/*/lib)
shopt -u nullglob
if [ "${#nvidia_lib_dirs[@]}" -gt 0 ]; then
    nvidia_ld_path=$(IFS=:; echo "${nvidia_lib_dirs[*]}")
    export LD_LIBRARY_PATH="${nvidia_ld_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$AURISDECK_DATA_DIR" "$HF_HOME"

if [ "$(id -u)" -eq 0 ] && [ "$PUID" != "0" ]; then
    if ! getent group "$PGID" >/dev/null 2>&1; then
        group_name="aurisdeck"
        if getent group "$group_name" >/dev/null 2>&1; then
            group_name="aurisdeck-${PGID}"
        fi
        groupadd -g "$PGID" "$group_name"
    fi

    if ! getent passwd "$PUID" >/dev/null 2>&1; then
        user_name="aurisdeck"
        if getent passwd "$user_name" >/dev/null 2>&1; then
            user_name="aurisdeck-${PUID}"
        fi
        useradd -u "$PUID" -g "$PGID" -M -s /usr/sbin/nologin -d /tmp "$user_name"
    fi

    target_owner="${PUID}:${PGID}"
    for target in \
        "$AURISDECK_DATA_DIR" \
        "$HF_HOME" \
        "$AURISDECK_DATA_DIR/profiles" \
        "$AURISDECK_DATA_DIR/glossaries" \
        "$AURISDECK_DATA_DIR/replacements" \
        "$AURISDECK_DATA_DIR/locales" \
        "$AURISDECK_DATA_DIR/media" \
        "$AURISDECK_DATA_DIR/checkpoints" \
        "$AURISDECK_DATA_DIR/outputs" \
        "$AURISDECK_DATA_DIR/tmp" \
        "$AURISDECK_DATA_DIR/aurisdeck.db" \
        "$AURISDECK_DATA_DIR/aurisdeck.db-wal" \
        "$AURISDECK_DATA_DIR/aurisdeck.db-shm" \
        "$AURISDECK_DATA_DIR/system_settings.json"; do
        if [ -e "$target" ] && [ "$(stat -c '%u:%g' "$target")" != "$target_owner" ]; then
            chown "$target_owner" "$target"
        fi
    done

    exec gosu "$target_owner" "$@"
fi

exec "$@"
