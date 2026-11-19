import mlx.core as mx


def quantize(w, bits, scheme="group", group_size=64, symmetric=True):
    w = w.astype(mx.float32)
    rows, cols = w.shape
    if scheme == "tensor":
        blocks = w.reshape(1, 1, -1)
    elif scheme == "row":
        blocks = w.reshape(rows, 1, cols)
    elif scheme == "group":
        assert cols % group_size == 0, "in_features must be divisible by group_size"
        blocks = w.reshape(rows, cols // group_size, group_size)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    if symmetric:
        qmin, qmax = -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
        offset = mx.zeros_like(blocks[..., :1])
        scale = mx.max(mx.abs(blocks), axis=-1, keepdims=True) / qmax
    else:
        qmin, qmax = 0, 2 ** bits - 1
        lo = mx.min(blocks, axis=-1, keepdims=True)
        hi = mx.max(blocks, axis=-1, keepdims=True)
        offset = lo
        scale = (hi - lo) / qmax
    scale = mx.where(scale == 0, 1.0, scale)
    q = mx.clip(mx.round((blocks - offset) / scale), qmin, qmax)
    q = q.astype(mx.int8 if symmetric else mx.uint8)
    return q, scale, offset, w.shape

def dequantize(q, scale, offset, shape):
    return (q.astype(mx.float32) * scale + offset).reshape(shape)

def reconstruction_error(w, w_hat):
    w = w.astype(mx.float32)
    err = w - w_hat
    rel = (mx.sqrt(mx.sum(err * err)) / mx.sqrt(mx.sum(w * w))).item()
    return rel, mx.mean(err * err).item(), mx.max(mx.abs(err)).item()

CONFIGS = [
    (8, "tensor", None, True,  "tensor"),
    (8, "row",    None, True,  "row"),
    (8, "group",  64,   True,  "group(64)"),
    (4, "tensor", None, True,  "tensor"),
    (4, "row",    None, True,  "row"),
    (4, "group",  64,   True,  "group(64)"),
    (4, "group",  32,   True,  "group(32)"),
    (4, "group",  64,   False, "asym g(64)"),
]

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "model.layers.12.mlp.down_proj.weight"
    w = mx.load("model.safetensors")[name]
    print(f"matrix: {name}  shape {tuple(w.shape)}  dtype {w.dtype}")
    print(f"{'bits':>4} {'scheme':>12} {'rel error':>10} {'mse':>10} {'max abs':>10} {'scales':>8}")
    for bits, scheme, gs, sym, label in CONFIGS:
        q, scale, offset, shape = quantize(w, bits, scheme, gs or 64, sym)
        rel, mse, mx_abs = reconstruction_error(w, dequantize(q, scale, offset, shape))
        print(f"{bits:>4} {label:>12} {rel:>10.4%} {mse:>10.2e} {mx_abs:>10.2e} {scale.size:>8}")
