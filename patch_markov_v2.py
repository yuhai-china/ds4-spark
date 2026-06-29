#!/usr/bin/env python3.12
"""Properly append Markov tensors to DSpark GGUF. Reads actual data sizes from
offset table — never guesses quant type sizes."""

import struct, os, json, sys
import numpy as np

HF = "/Users/yuhai/github/DeepSeek-V4-Flash-DSpark-4E"

def parse_gguf(path):
    with open(path, "rb") as f: data = f.read()
    assert data[:4] == b'GGUF'
    ver = struct.unpack_from('<I', data, 4)[0]
    ntensors = struct.unpack_from('<Q', data, 8)[0]
    nkv = struct.unpack_from('<Q', data, 16)[0]

    # Skip KVs
    pos = 24
    for _ in range(nkv):
        kl = struct.unpack_from('<Q', data, pos)[0]; pos += 8+kl
        t = struct.unpack_from('<I', data, pos)[0]; pos += 4
        if t == 8: sl = struct.unpack_from('<Q', data, pos)[0]; pos += 8+sl
        elif t == 9:
            et = struct.unpack_from('<I', data, pos)[0]; pos += 4
            n = struct.unpack_from('<Q', data, pos)[0]; pos += 8
            if et == 8:
                for _ in range(n):
                    sl = struct.unpack_from('<Q', data, pos)[0]; pos += 8+sl
            else:
                szs = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                pos += n*szs.get(et,4)
        elif t == 7: pos += 1
        else:
            szs = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,10:8,11:8,12:8}
            pos += szs.get(t,4)
    kv_end = pos

    tensors = []
    for i in range(ntensors):
        nl = struct.unpack_from('<Q', data, pos)[0]
        name = data[pos+8:pos+8+nl].decode().rstrip('\0')
        pos += 8+nl
        r = struct.unpack_from('<I', data, pos)[0]; pos += 4
        dims = [struct.unpack_from('<Q', data, pos+j*8)[0] for j in range(r)]
        pos += r*8
        tt = struct.unpack_from('<I', data, pos)[0]; pos += 4
        off = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        tensors.append((name, r, dims, tt, off))
    desc_end = pos
    return data, ver, nkv, kv_end, tensors, desc_end

def load_markov():
    with open(os.path.join(HF, "model.safetensors.index.json")) as f: idx = json.load(f)
    shard = "model-00048-of-00048.safetensors"
    result = {}
    with open(os.path.join(HF, shard), "rb") as f:
        hs = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(hs))
        for tname in ["mtp.2.markov_head.markov_w1.weight",
                       "mtp.2.markov_head.markov_w2.weight"]:
            m = hdr[tname]
            off = 8+hs+m["data_offsets"][0]; sz = m["data_offsets"][1]-m["data_offsets"][0]
            f.seek(off)
            raw = np.frombuffer(f.read(sz), dtype=np.uint16)
            u32 = raw.astype(np.uint32)<<16
            arr = np.frombuffer(u32.tobytes(), dtype=np.float32).copy()
            arr = arr.reshape(m["shape"][0], m["shape"][1]).T.copy()
            key = "w1" if "w1" in tname else "w2"
            result[key] = arr
            print(f"  {tname}: {m['shape']} -> GGUF {arr.shape} ({arr.nbytes/1024/1024:.1f} MiB)")
    return result["w1"], result["w2"]

def f32_to_f16(arr):
    f32 = np.asarray(arr, dtype=np.float32).ravel(); u32 = f32.view(np.uint32)
    s = (u32>>16)&0x8000; exp = (u32>>23)&0xff; man = (u32>>13)&0x3ff
    e = exp.astype(np.int32)-127+15; e = np.clip(e,0,31).astype(np.uint32)
    e[exp<113]=0; e[exp==0xff]=31; man[(exp==0xff)&(man==0)]=0
    return (s|(e<<10)|man).astype(np.uint16).tobytes()

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/yuhai/github/ds4/DeepSeek-V4-Flash-DSpark-4E.gguf"
    if not os.path.exists(gguf_path):
        print(f"ERROR: {gguf_path} not found"); sys.exit(1)

    print("Loading Markov tensors...")
    w1, w2 = load_markov()

    print("Parsing GGUF...")
    data, ver, nkv, kv_end, tensors, desc_end = parse_gguf(gguf_path)

    # Compute actual data sizes from offsets (pad-aligned)
    align = 32
    data_start = ((desc_end + align - 1)//align)*align
    for i in range(len(tensors)):
        name, r, dims, tt, off = tensors[i]
        if i+1 < len(tensors):
            sz = tensors[i+1][4] - off  # next offset - this offset
        else:
            sz = len(data) - data_start - off
        # Pad to alignment  
        padded_sz = ((sz + align - 1)//align)*align
        tensors[i] = (name, r, dims, tt, off, sz, padded_sz)

    # Compute where to place new tensors
    last = tensors[-1]
    new_off = last[4] + last[6]  # last offset + padded size
    new_off = ((new_off + align - 1)//align)*align

    # Add Markov tensors
    extras = []
    for ename, earr in [("markov_head.w1.weight", w1), ("markov_head.w2.weight", w2)]:
        f16buf = f32_to_f16(earr)
        eshape = list(earr.shape)
        esz = len(f16buf)
        padded_esz = ((esz + align - 1)//align)*align
        extras.append((ename, eshape, f16buf, esz, padded_esz))

    # Write new GGUF
    print("(no backup - overwriting in place)")

    with open(gguf_path, "wb") as f:
        f.write(b'GGUF')
        f.write(struct.pack('<I', ver))
        f.write(struct.pack('<Q', len(tensors) + len(extras)))
        f.write(struct.pack('<Q', nkv))
        f.write(data[24:kv_end])

        cur = 0
        for name, r, dims, tt, off, sz, padded_sz in tensors:
            b = name.encode()
            f.write(struct.pack('<Q', len(b))); f.write(b)
            f.write(struct.pack('<I', r))
            for d in dims: f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', tt))
            f.write(struct.pack('<Q', cur))
            cur += padded_sz

        for ename, eshape, ebuf, esz, padded_esz in extras:
            b = ename.encode()
            f.write(struct.pack('<Q', len(b))); f.write(b)
            f.write(struct.pack('<I', len(eshape)))
            for d in eshape: f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', 1))  # F16
            f.write(struct.pack('<Q', cur))
            cur += padded_esz

        p = f.tell(); al = ((p+align-1)//align)*align
        f.write(b'\0'*(al-p))

        # Existing data
        for name, r, dims, tt, off, sz, padded_sz in tensors:
            src = data_start + off
            f.write(data[src:src+sz])
            f.write(b'\0'*(padded_sz-sz))

        # Extra data
        for ename, eshape, ebuf, esz, padded_esz in extras:
            f.write(ebuf)
            f.write(b'\0'*(padded_esz-esz))

    print(f"Written: {gguf_path} ({len(tensors)+len(extras)} tensors)")

if __name__ == "__main__":
    main()
