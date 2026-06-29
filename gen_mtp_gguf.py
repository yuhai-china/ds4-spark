#!/usr/bin/env python3.12
"""Build MTP GGUF from DSpark checkpoint for ds4 speculative decoding.
Reads DSpark mtp.2 layer + base model shared tensors, dequantizes to F32/F16,
renames to ds4's mtp.0.* convention, writes a standalone GGUF.

Usage:
    python3.12 gen_mtp_gguf.py \
        --hf /path/to/DeepSeek-V4-Flash-DSpark-4E \
        --out /tmp/mtp.gguf
"""

import json, struct, os, argparse
import numpy as np

# ── Low-level GGUF helpers ──
def w_u32(f, v): f.write(struct.pack('<I', v))
def w_u64(f, v): f.write(struct.pack('<Q', v))
def w_f32(f, v): f.write(struct.pack('<f', v))
def w_f64(f, v): f.write(struct.pack('<d', v))
def w_bool(f, v): f.write(struct.pack('<?', v))
def w_string(f, s):
    b = s.encode()
    w_u64(f, len(b))
    f.write(b)

def align(f, a=32):
    pos = f.tell()
    al = ((pos + a - 1) // a) * a
    if al > pos: f.write(b'\0' * (al - pos))

# ── Safetensors helpers ──
def read_st(hf_dir, st_file):
    with open(os.path.join(hf_dir, st_file), "rb") as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(hs))
    return h, 8 + hs

def read_st_raw(hf_dir, st_file, name):
    h, off = read_st(hf_dir, st_file)
    m = h[name]
    sh = tuple(m["shape"])
    dt = m["dtype"]
    start = off + m["data_offsets"][0]
    size = m["data_offsets"][1] - m["data_offsets"][0]
    with open(os.path.join(hf_dir, st_file), "rb") as f:
        f.seek(start)
        raw = f.read(size)
    return raw, sh, dt

# ── Dequantization ──
FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

def dequant(raw, shape, dtype):
    n = 1
    for s in shape: n *= s
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    if dtype in ("BF16", "F16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.frombuffer(u32.tobytes(), dtype=np.float32).copy().reshape(shape)
    if dtype == "F8_E4M3":
        u8 = np.frombuffer(raw, dtype=np.uint8)
        out = np.zeros(n, dtype=np.float32)
        # Vectorized FP8 dequant
        abs_x = u8 & 0x7f
        sign = np.where(u8 & 0x80, -1.0, 1.0)
        exp = (u8 >> 3) & 0x0f
        man = u8 & 0x07
        is_zero = abs_x == 0
        is_nan = abs_x == 0x7f
        val = np.where(exp > 0, 1.0 + man / 8.0, man / 8.0)
        out = sign * val * np.where(exp > 0, 2.0 ** (exp.astype(np.float32) - 7), 2.0 ** -6)
        out[is_zero | is_nan] = 0.0
        return out.reshape(shape)
    if dtype == "F8_E8M0":
        u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
        u32 = np.where(u8 == 0, np.uint32(0x00400000), u8 << 23)
        return np.frombuffer(u32.astype(np.uint32).tobytes(), dtype=np.float32).copy().reshape(shape)
    if dtype == "I8":
        return raw  # packed FP4
    raise ValueError(f"unknown: {dtype}")

def load_f32(hf_dir, wm, name):
    raw, shape, dtype = read_st_raw(hf_dir, wm[name], name)
    return dequant(raw, shape, dtype)

def load_f32_opt(hf_dir, wm, name):
    """Load tensor if exists, return None otherwise."""
    if name not in wm: return None
    return load_f32(hf_dir, wm, name)

def dequant_fp4_experts(hf_dir, wm, prefix, n_expert=256):
    """Dequant stacked per-expert FP4 w2 weights into [n_ff_exp, n_embd, n_expert]."""
    n_embd, n_ff_exp = 4096, 2048
    # w2 in safetensors: [n_embd=4096, n_ff_exp_packed=1024] -> real [4096, 2048]
    # GGUF expects: [n_ff_exp=2048, n_embd=4096, n_expert=256]
    result = np.zeros((n_ff_exp, n_embd, n_expert), dtype=np.float32)

    for xid in range(n_expert):
        w_raw, w_shape, _ = read_st_raw(hf_dir, wm[f"{prefix}{xid}.w2.weight"],
                                         f"{prefix}{xid}.w2.weight")
        s_raw, s_shape, _ = read_st_raw(hf_dir, wm[f"{prefix}{xid}.w2.scale"],
                                         f"{prefix}{xid}.w2.scale")
        scales = dequant(s_raw, s_shape, "F8_E8M0").ravel()
        w_data = np.frombuffer(w_raw, dtype=np.uint8)
        rows, packed = w_shape  # [4096, 1024]

        layer = np.zeros((rows, packed*2), dtype=np.float32)  # [4096, 2048]
        for r in range(rows):
            blk = 0
            for c in range(packed):
                b = int(w_data[r * packed + c])
                s = scales[blk] if blk < len(scales) else 1.0
                layer[r, c*2]     = FP4_TABLE[b & 0x0f] * s
                layer[r, c*2 + 1] = FP4_TABLE[(b >> 4) & 0x0f] * s
                if (c + 1) % 16 == 0:
                    blk += 1
        # Transpose: [4096, 2048] → [2048, 4096] to match [n_ff_exp, n_embd]
        result[:, :, xid] = layer.T

        if (xid + 1) % 64 == 0:
            print(f"  down expert {xid+1}/{n_expert}")

    return result



def f32_to_f16(arr):
    """Convert float32 numpy array to float16 bytes (simplified)."""
    f32 = np.asarray(arr, dtype=np.float32).ravel()
    u32 = f32.view(np.uint32)
    sign = (u32 >> 16) & 0x8000
    exp = (u32 >> 23) & 0xff
    man = (u32 >> 13) & 0x3ff
    e = exp.astype(np.int32) - 127 + 15
    e = np.clip(e, 0, 31).astype(np.uint32)
    is_sub = exp < 113
    e[is_sub] = 0
    is_nan_inf = exp == 0xff
    e[is_nan_inf] = 31
    man[is_nan_inf & (man == 0)] = 0  # inf
    u16 = sign | (e << 10) | man
    return u16.astype(np.uint16).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="DSpark HF directory")
    ap.add_argument("--out", required=True, help="Output MTP GGUF path")
    args = ap.parse_args()

    hf = args.hf
    with open(os.path.join(hf, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    N_EMBD, N_VOCAB, N_FF_EXP, N_EXP, N_EXP_USED = 4096, 129280, 2048, 256, 6

    print(f"Loading token embedding...")
    embed = load_f32(hf, wm, "embed.weight")
    output = load_f32(hf, wm, "head.weight")  # shared with base

    print(f"Loading MTP proj/norm...")
    main_proj = load_f32(hf, wm, "mtp.0.main_proj.weight")  # [4096, 12288]
    e_proj = main_proj[:, :4096].copy()
    h_proj = main_proj[:, 4096:8192].copy()
    main_norm = load_f32(hf, wm, "mtp.0.main_norm.weight")
    mtp_norm = load_f32_opt(hf, wm, "mtp.2.norm.weight")
    if mtp_norm is None: mtp_norm = main_norm.copy()

    print(f"Loading MTP attention tensors...")
    attn = {}
    attn_names = {
        "attn_sinks.weight": "mtp.2.attn.attn_sink",
        "attn_q_a.weight": "mtp.2.attn.wq_a.weight",
        "attn_q_b.weight": "mtp.2.attn.wq_b.weight",
        "attn_q_a_norm.weight": "mtp.2.attn.q_norm.weight",
        "attn_kv.weight": "mtp.2.attn.wkv.weight",
        "attn_kv_a_norm.weight": "mtp.2.attn.kv_norm.weight",
        "attn_output_a.weight": "mtp.2.attn.wo_a.weight",
        "attn_output_b.weight": "mtp.2.attn.wo_b.weight",
        "attn_norm.weight": "mtp.2.attn_norm.weight",
    }
    for gguf_name, hf_name in attn_names.items():
        d = load_f32_opt(hf, wm, hf_name)
        if d is not None:
            attn[gguf_name] = d

    print(f"Loading MTP HC tensors...")
    hc = {}
    hc_names = {
        "hc_attn_fn.weight": "mtp.2.hc_attn_fn",
        "hc_attn_scale.weight": "mtp.2.hc_attn_scale",
        "hc_attn_base.weight": "mtp.2.hc_attn_base",
        "hc_ffn_fn.weight": "mtp.2.hc_ffn_fn",
        "hc_ffn_scale.weight": "mtp.2.hc_ffn_scale",
        "hc_ffn_base.weight": "mtp.2.hc_ffn_base",
        "hc_head_base.weight": "mtp.2.hc_head_base",
        "hc_head_fn.weight": "mtp.2.hc_head_fn",
        "hc_head_scale.weight": "mtp.2.hc_head_scale",
    }
    for gguf_name, hf_name in hc_names.items():
        d = load_f32_opt(hf, wm, hf_name)
        if d is not None:
            hc[gguf_name] = d

    print(f"Loading MTP FFN tensors...")
    ffn = {}
    ffn_names = {
        "ffn_norm.weight": "mtp.2.ffn_norm.weight",
        "ffn_gate_inp.weight": "mtp.2.ffn.gate.weight",
        "exp_probs_b.bias": "mtp.2.ffn.gate.bias",
        "ffn_gate_shexp.weight": "mtp.2.ffn.shared_experts.w1.weight",
        "ffn_up_shexp.weight": "mtp.2.ffn.shared_experts.w3.weight",
        "ffn_down_shexp.weight": "mtp.2.ffn.shared_experts.w2.weight",
    }
    for gguf_name, hf_name in ffn_names.items():
        d = load_f32_opt(hf, wm, hf_name)
        if d is not None:
            ffn[gguf_name] = d

    print(f"Dequantizing MTP routed experts (this takes ~2 min)...")
    down_exps = dequant_fp4_experts(hf, wm, "mtp.2.ffn.experts.")  # [n_ff_exp, n_embd, n_expert]
    print(f"  w1/w3 experts...")
    n_embd, n_ff_exp = 4096, 2048
    gate_exps = np.zeros((n_embd, n_ff_exp, N_EXP), dtype=np.float32)
    up_exps = np.zeros((n_embd, n_ff_exp, N_EXP), dtype=np.float32)
    for wpart, target in [("w1", gate_exps), ("w3", up_exps)]:
        for xid in range(N_EXP):
            w_raw, w_shape, _ = read_st_raw(hf, wm[f"mtp.2.ffn.experts.{xid}.{wpart}.weight"],
                                             f"mtp.2.ffn.experts.{xid}.{wpart}.weight")
            s_raw, s_shape, _ = read_st_raw(hf, wm[f"mtp.2.ffn.experts.{xid}.{wpart}.scale"],
                                             f"mtp.2.ffn.experts.{xid}.{wpart}.scale")
            scales = dequant(s_raw, s_shape, "F8_E8M0").ravel()
            w_data = np.frombuffer(w_raw, dtype=np.uint8)
            rows, packed = w_shape  # [2048, 2048] for [2048, 4096]
            layer = np.zeros((rows, packed*2), dtype=np.float32)
            for r in range(rows):
                blk = 0
                for c in range(packed):
                    b = int(w_data[r * packed + c])
                    s = scales[blk] if blk < len(scales) else 1.0
                    layer[r, c*2]     = FP4_TABLE[b & 0x0f] * s
                    layer[r, c*2 + 1] = FP4_TABLE[(b >> 4) & 0x0f] * s
                    if (c + 1) % 16 == 0:
                        blk += 1
            target[:, :, xid] = layer.T  # transpose: [2048, 4096] → [4096, 2048]
        print(f"  {wpart} done ({N_EXP} experts)")

    print(f"Building GGUF...")

    # Build tensor list: (name, data_f32, store_as_f16)
    # 1D tensors stay F32, 2D+ tensors store as F16
    ts = []
    def add(name, data, f16=True):
        if data is None:
            print(f"  SKIP {name} (not found)")
            return
        data = np.asarray(data, dtype=np.float32)
        is_1d = data.ndim == 1
        ts.append((name, data, f16 and not is_1d))

    add("mtp.0.token_embd.weight", embed, True)
    add("mtp.0.output.weight", output, True)

    add("mtp.0.e_proj.weight", e_proj, True)
    add("mtp.0.h_proj.weight", h_proj, True)
    add("mtp.0.enorm.weight", main_norm, False)
    add("mtp.0.hnorm.weight", main_norm, False)
    add("mtp.0.norm.weight", mtp_norm, False)

    for name, data in attn.items():
        add(f"mtp.0.{name}", data, True)
    for name, data in hc.items():
        add(f"mtp.0.{name}", data, True)
    for name, data in ffn.items():
        add(f"mtp.0.{name}", data, True)

    add("mtp.0.ffn_gate_exps.weight", gate_exps, True)
    add("mtp.0.ffn_up_exps.weight", up_exps, True)
    add("mtp.0.ffn_down_exps.weight", down_exps, True)

    print(f"Total tensors: {len(ts)}")

    # Metadata
    meta = [
        ("general.architecture", 8, "deepseek4"),
        ("deepseek4.block_count", 4, 1),
        ("deepseek4.embedding_length", 4, N_EMBD),
        ("deepseek4.vocab_size", 4, N_VOCAB),
        ("deepseek4.attention.head_count", 4, 64),
        ("deepseek4.attention.head_count_kv", 4, 1),
        ("deepseek4.attention.key_length", 4, 512),
        ("deepseek4.attention.value_length", 4, 512),
        ("deepseek4.rope.dimension_count", 4, 64),
        ("deepseek4.attention.q_lora_rank", 4, 1024),
        ("deepseek4.attention.output_lora_rank", 4, 1024),
        ("deepseek4.attention.output_group_count", 4, 8),
        ("deepseek4.expert_count", 4, N_EXP),
        ("deepseek4.expert_used_count", 4, N_EXP_USED),
        ("deepseek4.expert_feed_forward_length", 4, N_FF_EXP),
        ("deepseek4.expert_shared_count", 4, 1),
        ("deepseek4.hash_layer_count", 4, 3),
        ("deepseek4.attention.sliding_window", 4, 128),
        ("deepseek4.attention.indexer.head_count", 4, 64),
        ("deepseek4.attention.indexer.key_length", 4, 128),
        ("deepseek4.attention.indexer.top_k", 4, 512),
        ("deepseek4.hyper_connection.count", 4, 4),
        ("deepseek4.hyper_connection.sinkhorn_iterations", 4, 20),
        ("deepseek4.attention.layer_norm_rms_epsilon", 6, 1e-6),
        ("deepseek4.hyper_connection.epsilon", 6, 1e-6),
        ("deepseek4.expert_weights_scale", 6, 1.5),
        ("deepseek4.expert_weights_norm", 7, True),
        ("deepseek4.rope.freq_base", 12, 10000.0),
        ("deepseek4.rope.scaling.factor", 12, 16.0),
        ("deepseek4.rope.scaling.yarn_beta_fast", 6, 32.0),
        ("deepseek4.rope.scaling.yarn_beta_slow", 6, 1.0),
        ("deepseek4.rope.scaling.original_context_length", 10, 65536),
        ("deepseek4.swiglu_clamp_exp", 6, [10.0]),
        ("deepseek4.attention.compress_ratios", 4, [0]),
    ]

    # Write GGUF
    with open(args.out, "wb") as f:
        f.write(GGUF_MAGIC)
        w_u32(f, 3)  # version
        w_u64(f, len(ts))
        w_u64(f, len(meta))

        for key, tcode, val in meta:
            w_string(f, key)
            w_u32(f, tcode)
            if tcode == 4:  # UINT32
                if isinstance(val, list):
                    w_u32(f, 4); w_u64(f, len(val))
                    for v in val: w_u32(f, v)
                else:
                    w_u32(f, val)
            elif tcode == 6:  # FLOAT32
                if isinstance(val, list):
                    w_u32(f, 6); w_u64(f, len(val))
                    for v in val: w_f32(f, v)
                else:
                    w_f32(f, val)
            elif tcode == 7:
                w_bool(f, val)
            elif tcode == 8:
                w_string(f, str(val))
            elif tcode == 10:
                w_u64(f, val)
            elif tcode == 12:
                w_f64(f, val)

        # Tensor infos
        offset = 0
        for name, data, is_f16 in ts:
            w_string(f, name)
            ndim = data.ndim
            w_u32(f, ndim)
            for d in reversed(data.shape):
                w_u64(f, d)
            w_u32(f, 1 if is_f16 else 0)  # F16=1, F32=0
            w_u64(f, offset)
            n = data.size
            nbytes = n * (2 if is_f16 else 4)
            offset += nbytes
            offset = ((offset + 31) // 32) * 32

        align(f)

        # Tensor data
        for name, data, is_f16 in ts:
            if is_f16:
                f.write(f32_to_f16(data))
            else:
                f.write(data.astype(np.float32).tobytes())
            align(f)

    size_mb = os.path.getsize(args.out) / (1024*1024)
    size_gb = size_mb / 1024
    print(f"\nWrote: {args.out}")
    print(f"Size: {size_gb:.1f} GiB")
    print(f"\nTest with:")
    print(f"  ds4 --mtp {args.out} --mtp-draft 4 -p 'Hello' -n 100")

if __name__ == "__main__":
    main()
