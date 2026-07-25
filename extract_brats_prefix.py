"""
extract_brats_prefix.py

Stream-extract complete BraTS case files from a PARTIALLY-downloaded ZIP64
archive, without needing the (end-of-file) central directory. The Kaggle
archive is a ZIP64 with no data descriptors, so local file headers carry the
real compressed sizes (in the ZIP64 extra field) and we can walk entries
deterministically from the front, inflating each one whose data lies fully
inside the safely-downloaded contiguous prefix.

This is what lets us start training on ~80 BraTS cases without waiting for
the full 4.47GB on a slow link.
"""
import os
import struct
import sys
import zlib

ARCHIVE = r"C:/Projects/Yugma/data/brats_full/archive.zip"
OUT_ROOT = r"C:/Projects/Yugma/data/brats_subset"
# only [0, BOUNDARY) is guaranteed pure contiguous curl data
BOUNDARY = 1073741824  # 1 GiB

LFH = b'PK\x03\x04'


def decode_zip64_extra(extra: bytes, csize: int, usize: int):
    """Return real (usize, csize) from the ZIP64 extended-info extra field
    when the 32-bit fields hold the 0xFFFFFFFF sentinel."""
    i = 0
    while i + 4 <= len(extra):
        tag, size = struct.unpack_from('<HH', extra, i)
        i += 4
        if tag == 0x0001:
            vals = []
            off = i
            # order: uncompressed, compressed, then others - present only for
            # fields that were 0xFFFFFFFF
            if usize == 0xFFFFFFFF:
                vals.append(struct.unpack_from('<Q', extra, off)[0]); off += 8
            if csize == 0xFFFFFFFF:
                vals.append(struct.unpack_from('<Q', extra, off)[0]); off += 8
            real_u = vals[0] if usize == 0xFFFFFFFF else usize
            real_c = (vals[1] if (usize == 0xFFFFFFFF and csize == 0xFFFFFFFF)
                      else (vals[0] if csize == 0xFFFFFFFF else csize))
            return real_u, real_c
        i += size
    return usize, csize


def main():
    only_training = True
    f = open(ARCHIVE, 'rb')
    extracted = 0
    skipped_incomplete = 0
    cases = set()

    while True:
        pos = f.tell()
        sig = f.read(4)
        if sig != LFH:
            print(f"end of walkable entries at offset {pos} (sig={sig!r})")
            break
        hdr = f.read(26)
        if len(hdr) < 26:
            break
        ver, flag, method, mtime, mdate, crc, csize, usize, nlen, elen = \
            struct.unpack('<HHHHHIIIHH', hdr)
        name = f.read(nlen).decode('latin1')
        extra = f.read(elen)
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
            usize, csize = decode_zip64_extra(extra, csize, usize)

        data_start = f.tell()
        data_end = data_start + csize
        if data_end > BOUNDARY:
            print(f"  incomplete (data_end={data_end/1e6:.0f}MB > boundary): {os.path.basename(name)}")
            skipped_incomplete += 1
            break

        raw = f.read(csize)
        if len(raw) < csize:
            break

        keep = name.endswith('.nii') and (not only_training or 'TrainingData' in name)
        if keep:
            if method == 8:
                data = zlib.decompress(raw, -15)
            elif method == 0:
                data = raw
            else:
                print(f"  unknown method {method} for {name}, skip"); continue
            out_path = os.path.join(OUT_ROOT, name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as g:
                g.write(data)
            extracted += 1
            cases.add(os.path.dirname(name))

    print(f"\nextracted {extracted} files across {len(cases)} case folders "
          f"(skipped_incomplete={skipped_incomplete})")
    # report how many cases have the full set (4 modalities + seg)
    complete = 0
    for c in cases:
        d = os.path.join(OUT_ROOT, c)
        files = os.listdir(d) if os.path.isdir(d) else []
        has = sum(any(f.endswith(f"_{m}.nii") for f in files)
                  for m in ["flair", "t1", "t1ce", "t2", "seg"])
        if has == 5:
            complete += 1
    print(f"cases with all 5 files (4 modalities + seg): {complete}")


if __name__ == "__main__":
    main()
