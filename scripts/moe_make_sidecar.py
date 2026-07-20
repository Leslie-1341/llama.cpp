#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import math
import os
import struct
import sys


GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32

VT_UINT8 = 0
VT_INT8 = 1
VT_UINT16 = 2
VT_INT16 = 3
VT_UINT32 = 4
VT_INT32 = 5
VT_FLOAT32 = 6
VT_BOOL = 7
VT_STRING = 8
VT_ARRAY = 9
VT_UINT64 = 10
VT_INT64 = 11
VT_FLOAT64 = 12

MAGIC = b"LMOEBC1\0"
CODEC_RAW = 0
CODEC_GGML_QUANT = 1
CODEC_MWQ = 2
MWQ_BLOCK = 32

QK_K = 256
QUANT_SIZES = {
    0: (1, 4),                         # F32
    1: (1, 2),                         # F16
    2: (32, 2 + 16),                   # Q4_0
    3: (32, 2 + 2 + 16),               # Q4_1
    6: (32, 2 + 4 + 16),               # Q5_0
    7: (32, 2 + 2 + 4 + 16),           # Q5_1
    8: (32, 2 + 32),                   # Q8_0
    9: (32, 4 + 4 + 32),               # Q8_1
    10: (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    11: (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: (256, 2 + 2 + QK_K // 2 + 12), # Q4_K
    13: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    15: (256, 4 + QK_K + QK_K // 8),
    16: (256, 2 + QK_K // 4),
    17: (256, 2 + QK_K // 4 + QK_K // 32),
    18: (256, 2 + QK_K // 4 + QK_K // 8),
    19: (256, 2 + QK_K // 8 + QK_K // 16),
    20: (32, 2 + 16),
    21: (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: (256, 2 + QK_K // 4 + QK_K // 16),
    23: (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    29: (256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    30: (1, 2),
    34: (256, 2 + 4 * 13),
    35: (256, 2 + 64),
    39: (32, 16),
    40: (32, 16),
    41: (256, 2 + QK_K // 8),
}

GGML_TYPE_BY_BITS = {
    2: 10, # Q2_K
    3: 11, # Q3_K
    4: 12, # Q4_K
    5: 13, # Q5_K
    6: 14, # Q6_K
}


def align(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


class Reader:
    def __init__(self, path: str):
        self.f = open(path, "rb")

    def tell(self) -> int:
        return self.f.tell()

    def seek(self, off: int) -> None:
        self.f.seek(off)

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8")


def skip_value(r: Reader, typ: int):
    if typ in (VT_UINT8, VT_INT8, VT_BOOL):
        r.read(1)
    elif typ in (VT_UINT16, VT_INT16):
        r.read(2)
    elif typ in (VT_UINT32, VT_INT32, VT_FLOAT32):
        r.read(4)
    elif typ in (VT_UINT64, VT_INT64, VT_FLOAT64):
        r.read(8)
    elif typ == VT_STRING:
        r.read(r.u64())
    elif typ == VT_ARRAY:
        elem_typ = r.u32()
        n = r.u64()
        for _ in range(n):
            skip_value(r, elem_typ)
    else:
        raise ValueError(f"unknown GGUF value type {typ}")


def read_value(r: Reader, typ: int):
    if typ == VT_UINT32:
        return r.u32()
    if typ == VT_STRING:
        return r.string()
    skip_value(r, typ)
    return None


def tensor_nbytes(shape: list[int], typ: int) -> int:
    if typ not in QUANT_SIZES:
        raise ValueError(f"unsupported tensor type {typ}")
    block, type_size = QUANT_SIZES[typ]
    row = math.ceil(shape[0] / block) * type_size
    rest = 1
    for d in shape[1:]:
        rest *= d
    return row * rest


def f16(b: bytes) -> float:
    return struct.unpack("<e", b)[0]


def f16_pack(x: float) -> bytes:
    return struct.pack("<e", x)


def get_scale_min_k4(j: int, scales: bytes) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)


def dequant_q8_0(blob: bytes, n: int) -> array.array:
    out = array.array("f")
    out_extend = out.extend
    for off in range(0, len(blob), 34):
        d = f16(blob[off:off + 2])
        qs = blob[off + 2:off + 34]
        out_extend((struct.unpack("b", qs[i:i + 1])[0] * d for i in range(32)))
    if len(out) != n:
        raise RuntimeError(f"q8_0 dequant length mismatch {len(out)} != {n}")
    return out


def dequant_q5_0(blob: bytes, n: int) -> array.array:
    out = array.array("f")
    for off in range(0, len(blob), 22):
        d = f16(blob[off:off + 2])
        qh = struct.unpack("<I", blob[off + 2:off + 6])[0]
        qs = blob[off + 6:off + 22]
        vals = [0.0] * 32
        for j in range(16):
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            vals[j] = (((qs[j] & 0x0F) | xh0) - 16) * d
            vals[j + 16] = (((qs[j] >> 4) | xh1) - 16) * d
        out.extend(vals)
    if len(out) != n:
        raise RuntimeError(f"q5_0 dequant length mismatch {len(out)} != {n}")
    return out


def dequant_q4_k(blob: bytes, n: int) -> array.array:
    out = array.array("f")
    for off in range(0, len(blob), 144):
        d = f16(blob[off:off + 2])
        dmin = f16(blob[off + 2:off + 4])
        scales = blob[off + 4:off + 16]
        qs = blob[off + 16:off + 144]
        isub = 0
        for qoff in range(0, 128, 32):
            sc1, m1 = get_scale_min_k4(isub, scales)
            sc2, m2 = get_scale_min_k4(isub + 1, scales)
            d1, min1 = d * sc1, dmin * m1
            d2, min2 = d * sc2, dmin * m2
            q = qs[qoff:qoff + 32]
            out.extend(((x & 0x0F) * d1 - min1 for x in q))
            out.extend(((x >> 4) * d2 - min2 for x in q))
            isub += 2
    if len(out) != n:
        raise RuntimeError(f"q4_K dequant length mismatch {len(out)} != {n}")
    return out


def dequantize(blob: bytes, typ: int, n: int) -> array.array:
    if typ == 8:
        return dequant_q8_0(blob, n)
    if typ == 6:
        return dequant_q5_0(blob, n)
    if typ == 12:
        return dequant_q4_k(blob, n)
    raise RuntimeError(f"MWQ generator does not yet support source GGUF tensor type {typ}")


def mwq_encode(vals: array.array, bits: int) -> bytes:
    if bits < 1 or bits > 8:
        raise RuntimeError("MWQ bits must be in the range 1..8")

    qmax = (1 << bits) - 1
    qbytes = (MWQ_BLOCK * bits + 7) // 8
    out = bytearray()
    for off in range(0, len(vals), MWQ_BLOCK):
        block = vals[off:off + MWQ_BLOCK]
        if len(block) < MWQ_BLOCK:
            block.extend([0.0] * (MWQ_BLOCK - len(block)))
        mn = min(block)
        mx = max(block)
        scale = (mx - mn) / qmax if mx > mn else 0.0
        inv = (1.0 / scale) if scale != 0.0 else 0.0
        out += f16_pack(mn)
        out += f16_pack(scale)
        packed = bytearray(qbytes)
        for i, x in enumerate(block):
            q = int(round((x - mn) * inv)) if scale != 0.0 else 0
            q = 0 if q < 0 else qmax if q > qmax else q
            bit = i * bits
            byte = bit >> 3
            shift = bit & 7
            packed[byte] |= (q << shift) & 0xFF
            if shift + bits > 8:
                packed[byte + 1] |= q >> (8 - shift)
        out += packed
    return bytes(out)


def is_expert_tensor(name: str, shape: list[int]) -> bool:
    return "_exps" in name and len(shape) == 3 and shape[2] > 1


def parse_gguf(path: str):
    r = Reader(path)
    if r.u32() != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    _version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()

    alignment = GGUF_DEFAULT_ALIGNMENT
    for _ in range(kv_count):
        key = r.string()
        typ = r.u32()
        val = read_value(r, typ)
        if key == "general.alignment" and isinstance(val, int):
            alignment = val

    infos = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        shape = [r.u64() for _ in range(n_dims)]
        typ = r.u32()
        rel_off = r.u64()
        infos.append((name, shape, typ, rel_off))

    data_base = align(r.tell(), alignment)
    return [(name, shape, typ, data_base + rel_off, tensor_nbytes(shape, typ)) for name, shape, typ, rel_off in infos]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a MoE expert sidecar for llama-moe-buffer.")
    ap.add_argument("model", help="source GGUF model to extract expert slices from")
    ap.add_argument("output", help="sidecar output path")
    ap.add_argument("--bits", type=int, default=4, help="bits key to store in the sidecar")
    ap.add_argument("--block", type=int, default=32, help="MWQ block size for --codec mwq, normally 32/64/128")
    ap.add_argument("--v2", action="store_true", help="emit MWQ V2 hierarchical-scale sidecar when --codec mwq is used")
    ap.add_argument("--scale-group", type=int, default=16, help="MWQ V2 scale group size in blocks")
    ap.add_argument("--outliers", type=int, default=0, help="MWQ V2 max sparse residual outliers per block")
    ap.add_argument("--outlier-threshold", type=float, default=6.0, help="MWQ V2 z-score threshold for outlier residuals")
    ap.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="BITS:MODEL",
        help="add another precision source to the same sidecar, for example --add 3:/tmp/model-q3.gguf",
    )
    ap.add_argument(
        "--codec",
        choices=("raw", "ggml-quant", "mwq"),
        default="raw",
        help="raw copies bytes as-is; ggml-quant transcodes GGML low-bit slices; mwq builds variable-block multi-precision slices directly",
    )
    ap.add_argument(
        "--target-model",
        help="model whose expert slice size/layout will be decoded into; required when source precision differs",
    )
    args = ap.parse_args()

    if args.codec == "mwq":
        tool = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "bin", "llama-moe-sidecar"))
        if os.path.exists(tool):
            os.execv(tool, [tool] + sys.argv[1:])
        raise RuntimeError("MWQ sidecar generation is implemented by build/bin/llama-moe-sidecar; run: cmake --build build --target llama-moe-sidecar -j2")

    tensors = parse_gguf(args.model)
    target_by_name = {}
    if args.target_model:
        target_by_name = {name: (shape, typ, n_bytes) for name, shape, typ, _off, n_bytes in parse_gguf(args.target_model)}

    codec = {"raw": CODEC_RAW, "ggml-quant": CODEC_GGML_QUANT, "mwq": CODEC_MWQ}[args.codec]
    if codec == CODEC_GGML_QUANT:
        if args.bits not in GGML_TYPE_BY_BITS:
            raise RuntimeError(f"ggml-quant sidecars only support bits keys {sorted(GGML_TYPE_BY_BITS)}")
    if codec == CODEC_MWQ and (args.bits < 1 or args.bits > 8):
        raise RuntimeError("MWQ currently supports bits 1..8")

    sources = [(args.bits, args.model)]
    for item in args.add:
        bits_s, sep, path = item.partition(":")
        if not sep or not bits_s or not path:
            raise RuntimeError(f"invalid --add value {item!r}; expected BITS:MODEL")
        bits = int(bits_s)
        if codec == CODEC_GGML_QUANT and bits not in GGML_TYPE_BY_BITS:
            raise RuntimeError(f"ggml-quant sidecars only support bits keys {sorted(GGML_TYPE_BY_BITS)}")
        if codec == CODEC_MWQ and (bits < 1 or bits > 8):
            raise RuntimeError("MWQ currently supports bits 1..8")
        sources.append((bits, path))

    entries = []
    for bits, model in sources:
        source_tensors = tensors if model == args.model else parse_gguf(model)
        expected_type = GGML_TYPE_BY_BITS.get(bits)
        with open(model, "rb") as f:
            for name, shape, _typ, data_offset, n_bytes in source_tensors:
                if not is_expert_tensor(name, shape):
                    continue
                if codec == CODEC_GGML_QUANT and _typ != expected_type:
                    raise RuntimeError(f"{name} in {model} has type {_typ}, but bits {bits} expects GGUF type {expected_type}")
                n_expert = shape[2]
                if n_bytes % n_expert != 0:
                    raise RuntimeError(f"tensor {name} has non-uniform expert slices")
                stride = n_bytes // n_expert
                decoded_stride = stride
                if target_by_name:
                    if name not in target_by_name:
                        raise RuntimeError(f"target model is missing tensor {name}")
                    target_shape, _target_typ, target_n_bytes = target_by_name[name]
                    if target_shape != shape:
                        raise RuntimeError(f"target tensor {name} shape mismatch: {target_shape} != {shape}")
                    if target_n_bytes % n_expert != 0:
                        raise RuntimeError(f"target tensor {name} has non-uniform expert slices")
                    decoded_stride = target_n_bytes // n_expert
                for expert in range(n_expert):
                    f.seek(data_offset + expert * stride)
                    blob = f.read(stride)
                    if len(blob) != stride:
                        raise RuntimeError(f"short read for {name} expert {expert}")
                    if codec == CODEC_MWQ:
                        n_elem = math.prod(shape[:2])
                        blob = mwq_encode(dequantize(blob, _typ, n_elem), bits)
                    entries.append((name, expert, bits, codec, decoded_stride, len(blob), blob))

    header_size = 16
    index_size = sum(34 + len(name.encode("utf-8")) for name, *_ in entries)
    blob_off = header_size + index_size
    records = []
    blobs = []
    for name, expert, bits, codec, decoded_size, encoded_size, blob in entries:
        name_b = name.encode("utf-8")
        records.append((name_b, expert, bits, codec, decoded_size, encoded_size, blob_off))
        blobs.append(blob)
        blob_off += encoded_size

    with open(args.output, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", len(records), 0))
        for name_b, expert, bits, codec, decoded_size, encoded_size, off in records:
            out.write(struct.pack("<H I H H Q Q Q", len(name_b), expert, bits, codec, decoded_size, encoded_size, off))
            out.write(name_b)
        for blob in blobs:
            out.write(blob)

    total = sum(len(b) for b in blobs)
    print(f"wrote {len(entries)} expert slices, {total / 1048576:.1f} MiB {args.codec} sidecar: {args.output}")


if __name__ == "__main__":
    main()
