Support DeepSeek V4 Flash 4Expert (top-4) with full GGUF conversion pipeline

This PR enables ds4 to run the 4Expert variant of DeepSeek V4 Flash, which routes
to top-4 experts instead of the default top-6.

## Motivation

DeepSeek V4 Flash comes in variants that activate different numbers of routed
experts per token. The original ds4 hardcodes 6 active experts. With this PR,
the 4Expert variant (256 total experts, 4 active per token) is supported out of
the box, while 6-expert models remain fully backward compatible.

4Expert model:     https://huggingface.co/cloudyu/DeepSeek-V4-Flash-4Expert
Pre-quantized GGUF: https://huggingface.co/cloudyu/DeepSeek-V4-Flash-4Expert-GGUF

## Changes

### ds4.c — 4Expert support with backward compatibility

- `DS4_SHAPE_FLASH.n_expert_used` changed from 6 to 4
- `g_ds4_shape.n_expert_used` changed from 6 to 4
- `ds4_select_shape_from_metadata()`: when matching Flash variant, accepts both
  4 and 6 as valid `n_expert_used` values. If 6 is detected (old GGUF),
  `g_ds4_shape.n_expert_used` is set to 6 at runtime. This ensures existing
  6-expert GGUF files continue to work without modification.

### gguf-tools/gen_gguf_template.py — New template generator

A Python 3 script that generates GGUF templates from safetensors metadata:

1. Reads the safetensors index for tensor names/shapes/dtypes
2. Maps HF tensor names to GGUF names using the same `layer_map` as
   `deepseek4-quantize.c`
3. Writes a template GGUF containing metadata, tokenizer data, and tensor
   descriptors (no weight data)
4. Automatically handles I64→I32 conversion for the tid2eid routing table

This replaces the workflow of hand-crafting GGUF templates. The template is
then fed to the existing `deepseek4-quantize` quantizer, which reads actual
weights from safetensors, applies the quantization policy, and produces the
final GGUF.

### docs/gguf-conversion.md — Conversion guide

Step-by-step guide for the full pipeline:
1. Generate template from safetensors
2. Quantize with layer-specific policy (Q4_K experts, Q8_0 projections,
   F16 attention/embedding)
3. Test inference

### docs/test-pr-on-linux.md — Linux testing guide

Two paths for testing on a fresh Linux machine:
- **Option A**: Full pipeline — clone, build, generate template, quantize, run
- **Option B**: Quick test — download pre-quantized GGUF and run directly

## Testing

- Q4_K GGUF (~153 GiB) converted from 4Expert safetensors using the template
  pipeline
- Successfully loads and runs with ds4 at ~26.70 t/s
- Existing 6-expert GGUF files continue to work (shape auto-detection preserves
  n_expert_used=6)
- All changes compile cleanly on macOS with `make`

## Build

```bash
# macOS:
make -j$(sysctl -n hw.ncpu)
make -C gguf-tools -j$(sysctl -n hw.ncpu)

# Linux:
make cpu -j$(nproc)
make -C gguf-tools -j$(nproc)
```
