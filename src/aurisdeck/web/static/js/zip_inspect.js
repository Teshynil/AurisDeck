/**
 * Client-side ZIP Central Directory inspector using File.slice() + DataView.
 * Reads only the trailing metadata of a local .zip file without uploading it.
 */

const MEDIA_EXTENSIONS = new Set([
  ".wav",
  ".flac",
  ".mp3",
  ".ogg",
  ".opus",
  ".m4a",
  ".aac",
  ".mp4",
  ".mov",
  ".webm",
  ".mkv",
]);

export function supportsLocalZipInspection() {
  return (
    typeof Blob !== "undefined" &&
    typeof Blob.prototype?.arrayBuffer === "function" &&
    typeof DataView === "function" &&
    typeof TextDecoder === "function"
  );
}

function isIgnoredMember(normalizedPath) {
  if (!normalizedPath || normalizedPath.endsWith("/")) {
    return true;
  }
  const parts = normalizedPath.split("/").filter(Boolean);
  if (parts.length === 0) {
    return true;
  }
  if (parts[0] === "__MACOSX") {
    return true;
  }
  const basename = parts[parts.length - 1];
  return basename.startsWith(".") || basename === "Thumbs.db";
}

function getExtension(filename) {
  const idx = filename.lastIndexOf(".");
  if (idx <= 0) {
    return "";
  }
  return filename.slice(idx).toLowerCase();
}

function readUint64(view, offset) {
  const low = view.getUint32(offset, true);
  const high = view.getUint32(offset + 4, true);
  return high * 4294967296 + low;
}

/**
 * Inspect a local File (.zip) and return supported media entries sorted by path.
 * @param {File} file
 * @returns {Promise<{ entries: Array<{ filename: string, path: string }>, hasHints: boolean }>}
 */
export async function inspectLocalZipFile(file) {
  if (!supportsLocalZipInspection()) {
    throw new Error("Local ZIP inspection is not supported in this browser");
  }
  const size = Number(file?.size || 0);
  if (size < 22) {
    throw new Error("Invalid ZIP archive");
  }

  const tailSize = Math.min(size, 65557);
  const tailStart = size - tailSize;
  const tailBuf = await file.slice(tailStart, size).arrayBuffer();
  const tailView = new DataView(tailBuf);

  let eocdOffset = -1;
  for (let i = tailSize - 22; i >= 0; i--) {
    if (tailView.getUint32(i, true) === 0x06054b50) {
      eocdOffset = i;
      break;
    }
  }
  if (eocdOffset === -1) {
    throw new Error("Invalid ZIP archive: End of Central Directory not found");
  }

  let cdSize = tailView.getUint32(eocdOffset + 12, true);
  let cdOffset = tailView.getUint32(eocdOffset + 16, true);

  // Check for ZIP64 EOCD Locator immediately preceding EOCD
  if ((cdSize === 0xffffffff || cdOffset === 0xffffffff) && eocdOffset >= 20) {
    if (tailView.getUint32(eocdOffset - 20, true) === 0x07064b50) {
      const zip64EocdOffset = readUint64(tailView, eocdOffset - 12);
      if (zip64EocdOffset + 56 <= size) {
        const z64Buf = await file.slice(zip64EocdOffset, zip64EocdOffset + 56).arrayBuffer();
        const z64View = new DataView(z64Buf);
        if (z64View.getUint32(0, true) === 0x06064b50) {
          cdSize = readUint64(z64View, 40);
          cdOffset = readUint64(z64View, 48);
        }
      }
    }
  }

  if (cdOffset + cdSize > size || cdSize === 0) {
    throw new Error("Invalid ZIP archive: corrupt Central Directory bounds");
  }

  const cdBuf = await file.slice(cdOffset, cdOffset + cdSize).arrayBuffer();
  const cdView = new DataView(cdBuf);
  const decoder = new TextDecoder("utf-8", { fatal: false });

  const entries = [];
  let hasHints = false;
  let pos = 0;

  while (pos + 46 <= cdBuf.byteLength) {
    if (cdView.getUint32(pos, true) !== 0x02014b50) {
      break;
    }
    const nameLen = cdView.getUint16(pos + 28, true);
    const extraLen = cdView.getUint16(pos + 30, true);
    const commentLen = cdView.getUint16(pos + 32, true);
    const nameStart = pos + 46;
    const nameEnd = nameStart + nameLen;
    if (nameEnd > cdBuf.byteLength) {
      break;
    }

    const rawName = decoder.decode(new Uint8Array(cdBuf, nameStart, nameLen));
    const normalized = rawName.replaceAll("\\", "/");
    pos = nameEnd + extraLen + commentLen;

    if (isIgnoredMember(normalized)) {
      continue;
    }
    const basename = normalized.split("/").pop() || "";
    const lowered = basename.toLowerCase();
    if (lowered === "info.txt" || lowered.endsWith(".aup")) {
      hasHints = true;
      continue;
    }
    if (MEDIA_EXTENSIONS.has(getExtension(lowered))) {
      entries.push({ filename: basename, path: normalized });
    }
  }

  entries.sort((a, b) => a.path.localeCompare(b.path));
  return { entries, hasHints };
}
