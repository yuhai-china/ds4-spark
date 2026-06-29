#!/usr/bin/env python3.12
"""Build MTP safetensors from DeepSeek-V4-Flash-DSpark-4E model.

Extracts ALL mtp.2.* tensors plus embed.weight, head.weight, layers.0.ffn.gate.tid2eid
from the base model, maps them to Flash-compatible names, and writes a single
model.safetensors at /tmp/mtp-sf-tmp/.

Does NOT dequantize – copies raw bytes directly.
"""

import json
import os
import shutil
import struct
from collections import defaultdict

SRC_DIR = "/Users/yuhai/github/DeepSeek-V4-Flash-DSpark-4E"
OUT_DIR = "/tmp/mtp-sf-tmp"


def _map_name(src_name: str) -> str | None:
    """Map a source tensor name to a Flash-compatible output name.

    Returns None if this tensor should be skipped.
    """
    if "mtp.2" not in src_name:
        # Base model tensors – keep as-is, except we exclude norm.weight
        # because mtp.2.norm.weight replaces it.
        if src_name == "norm.weight":
            return None
        return src_name

    # ── Handle mtp.2.* tensors ──

    if src_name == "mtp.2.norm.weight":
        return "norm.weight"

    if src_name.startswith("mtp.2.hc_head_"):
        return src_name.replace("mtp.2.", "")

    if src_name.startswith("mtp.2.ffn.experts."):
        return src_name.replace("mtp.2.", "layers.0.")

    # Everything else: mtp.2.xxx → layers.0.xxx
    return src_name.replace("mtp.2.", "layers.0.")


def _read_safetensors_tensors(st_path: str, wanted: list[str]) -> dict[str, tuple]:
    """Read specific tensors from a single safetensors file.

    Returns: {tensor_name: (dtype, shape, raw_bytes)}
    """
    with open(st_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode())
        data_start = 8 + header_size

    result = {}
    for name in wanted:
        if name not in header:
            print(f"  WARNING: {name} not found in {os.path.basename(st_path)}")
            continue
        info = header[name]
        start, end = info["data_offsets"]
        with open(st_path, "rb") as f:
            f.seek(data_start + start)
            raw = f.read(end - start)
        result[name] = (info["dtype"], tuple(info["shape"]), raw)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load weight map ──
    index_path = os.path.join(SRC_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # ── 2. Build source-to-destination mapping ──
    # source_tensor_name -> (dest_name, source_safetensors_file)
    to_extract: dict[str, tuple[str, str]] = {}  # src -> (dst, file)

    # Base model tensors we always want
    for src_name in ["embed.weight", "head.weight", "layers.0.ffn.gate.tid2eid"]:
        if src_name in weight_map:
            to_extract[src_name] = (src_name, weight_map[src_name])

    # ALL mtp.2.* tensors
    mtp2_count = 0
    for src_name, src_file in weight_map.items():
        if "mtp.2" not in src_name:
            continue
        dst = _map_name(src_name)
        if dst is None:
            continue
        mtp2_count += 1
        to_extract[src_name] = (dst, src_file)

    print(f"Source tensors to extract: {len(to_extract)} (from {mtp2_count} mtp.2 + {len(to_extract) - mtp2_count} base)")

    # ── 3. Group by source safetensors file ──
    by_file: dict[str, list[str]] = defaultdict(list)
    for src_name, (dst_name, src_file) in to_extract.items():
        by_file[src_file].append(src_name)

    # ── 4. Read tensors ──
    tensor_data: dict[str, tuple] = {}  # dst_name -> (dtype, shape, raw_bytes)
    for src_file, src_names in sorted(by_file.items()):
        st_path = os.path.join(SRC_DIR, src_file)
        print(f"Reading {len(src_names)} tensors from {src_file}...")
        raw_tensors = _read_safetensors_tensors(st_path, src_names)
        for src_name, (dtype, shape, raw) in raw_tensors.items():
            dst_name = to_extract[src_name][0]
            tensor_data[dst_name] = (dtype, shape, raw)

    # ── 5. Write single model.safetensors ──
    header: dict[str, dict] = {}
    offset = 0
    for dst_name in sorted(tensor_data):
        dtype, shape, raw = tensor_data[dst_name]
        end = offset + len(raw)
        header[dst_name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end

    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode()

    out_st = os.path.join(OUT_DIR, "model.safetensors")
    total_size = 0
    with open(out_st, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for dst_name in sorted(tensor_data):
            raw = tensor_data[dst_name][2]
            f.write(raw)
            total_size += len(raw)

    size_mb = total_size / (1024 * 1024)
    print(f"\nWritten {len(header)} tensors ({size_mb:.1f} MB) to {out_st}")

    # ── 6. Write index.json ──
    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: "model.safetensors" for name in sorted(tensor_data)},
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)

    # ── 7. Copy tokenizer files ──
    for fname in ["tokenizer_config.json", "tokenizer.json"]:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(OUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname}")
        else:
            print(f"WARNING: {fname} not found in source")

    # ── 8. Write config.json ──
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "qk_rope_head_dim": 64,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "vocab_size": 129280,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "hc_eps": 1e-06,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "num_hash_layers": 3,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "sliding_window": 128,
        "swiglu_limit": 10.0,
        "rope_theta": 10000,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "compress_rope_theta": 160000,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "max_position_embeddings": 1048576,
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── 9. Summary ──
    print(f"\n{'='*60}")
    print(f"Build complete.")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Tensors:    {len(header)}")
    print(f"  Files:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fname:<40} {mb:>8.1f} MB")


if __name__ == "__main__":
    main()
