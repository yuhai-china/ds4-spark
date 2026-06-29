#!/usr/bin/env python3.12
"""Build standalone Markov head GGUF from DSpark checkpoint.
Produces a small GGUF (~252 MiB F16) with just markov_w1 and markov_w2.

Usage: python3.12 gen_markov_gguf.py --hf <dir> --out markov.gguf
"""

import struct, os, json, argparse
import numpy as np

HF = "/Users/yuhai/github/DeepSeek-V4-Flash-DSpark-4E"

def load_markov(hf_dir):
    with open(os.path.join(hf_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    shard = "model-00048-of-00048.safetensors"
    result = {}
    with open(os.path.join(hf_dir, shard), "rb") as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hs))
        for tname in ["mtp.2.markov_head.markov_w1.weight",
                       "mtp.2.markov_head.markov_w2.weight"]:
            m = hdr[tname]
            off = 8+hs+m["data_offsets"][0]
            sz = m["data_offsets"][1]-m["data_offsets"][0]
            f.seek(off)
            raw = np.frombuffer(f.read(sz), dtype=np.uint16)
            u32 = raw.astype(np.uint32)<<16
            arr = np.frombuffer(u32.tobytes(), dtype=np.float32).copy()
            arr = arr.reshape(m["shape"][0], m["shape"][1]).T.copy()
            key = "w1" if "w1" in tname else "w2"
            result[key] = arr
    return result["w1"], result["w2"]

def f32_to_f16(arr):
    f32 = np.asarray(arr, dtype=np.float32).ravel()
    u32 = f32.view(np.uint32)
    s = (u32>>16)&0x8000; exp = (u32>>23)&0xff; man = (u32>>13)&0x3ff
    e = exp.astype(np.int32)-127+15; e = np.clip(e,0,31).astype(np.uint32)
    e[exp<113]=0; e[exp==0xff]=31; man[(exp==0xff)&(man==0)]=0
    return (s|(e<<10)|man).astype(np.uint16).tobytes()

def w_str(f, s):
    b = s.encode(); f.write(struct.pack('<Q', len(b))); f.write(b)
def w_u32(f, v): f.write(struct.pack('<I', v))
def w_u64(f, v): f.write(struct.pack('<Q', v))
def pad(f, a=32):
    p = f.tell(); al = ((p+a-1)//a)*a
    if al>p: f.write(b'\0'*(al-p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w1, w2 = load_markov(args.hf)
    w1_f16 = f32_to_f16(w1)
    w2_f16 = f32_to_f16(w2)

    rank = w1.shape[0]
    vocab = w1.shape[1]
    tensors = [
        ("markov_head.w1.weight", list(w1.shape), w1_f16, 1),
        ("markov_head.w2.weight", list(w2.shape), w2_f16, 1),
    ]

    with open(args.out, "wb") as f:
        f.write(b'GGUF')
        w_u32(f, 3)  # version
        w_u64(f, len(tensors))
        w_u64(f, 0)  # no KV metadata

        # Tensor headers
        cur = 0
        for name, shape, data_buf, ttype in tensors:
            w_str(f, name)
            w_u32(f, len(shape))
            for d in reversed(shape): w_u64(f, d)
            w_u32(f, ttype)
            w_u64(f, cur)
            cur += len(data_buf)
            cur = ((cur+31)//32)*32

        pad(f)

        # Tensor data
        for name, shape, data_buf, ttype in tensors:
            f.write(data_buf)
            padded = ((len(data_buf)+31)//32)*32
            f.write(b'\0'*(padded-len(data_buf)))

    size_mb = os.path.getsize(args.out)/1024/1024
    print(f"Written: {args.out} ({size_mb:.0f} MiB, {len(tensors)} tensors)")

if __name__ == "__main__":
    main()
