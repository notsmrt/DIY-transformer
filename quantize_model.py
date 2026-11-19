import sys
import mlx.core as mx
import kv_cache_model as km
import metrics
from quantize import quantize, dequantize

LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def fake_quantize(weights, bits, group_size=64, symmetric=True):
    out = dict(weights)
    count = 0
    for name, w in weights.items():
        if name.endswith(".weight") and any(f".{l}." in name for l in LINEAR):
            q, scale, offset, shape = quantize(w, bits, "group", group_size, symmetric)
            out[name] = dequantize(q, scale, offset, shape).astype(w.dtype)
            mx.eval(out[name])
            count += 1
    return out, count

def print_residual_diagnostics(label, ids, weights, layer_count=4, top_k=8):
    positions = [0, min(10, len(ids) - 1)]
    if positions[1] == positions[0] and len(ids) > 1:
        positions[1] = 1
    positions = list(dict.fromkeys(positions))

    x = weights["model.embed_tokens.weight"][mx.array(ids)]
    print(f"\n{label} residual stream (shape per position: {tuple(x.shape[1:])})")
    print(f"comparing token positions 0 and {positions[-1]} after each of the first {layer_count} layers")

    for layer in range(min(layer_count, km.config.n_layers)):
        normed = km.rms_norm(x, weights[f"model.layers.{layer}.input_layernorm.weight"])
        attention, _, _ = km.attention(normed, weights, layer, prefill=True)
        x = x + attention
        normed = km.rms_norm(x, weights[f"model.layers.{layer}.post_attention_layernorm.weight"])
        x = x + km.mlp(normed, weights, layer)
        mx.eval(x)

        print(f"  after layer {layer + 1}:")
        for position in positions:
            values = x[position].astype(mx.float32).tolist()
            rms = (sum(value * value for value in values) / len(values)) ** 0.5
            largest = sorted(enumerate(values), key=lambda item: abs(item[1]), reverse=True)[:top_k]
            dimensions = ", ".join(f"{index}={value:.3g}" for index, value in largest)
            max_abs = abs(largest[0][1]) if largest else 0.0
            print(
                f"    position {position}: rms={rms:.4g}, max_abs={max_abs:.4g}; "
                f"largest dims: {dimensions}"
            )

if __name__ == "__main__":
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    group_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    symmetric = not (len(sys.argv) > 3 and sys.argv[3] == "asym")
    tag = f"q{bits}g{group_size}" + ("" if symmetric else "a")

    qweights, count = fake_quantize(km.weights, bits, group_size, symmetric)
    print(f"quantized {count} linear weights -> tag {tag}")
    diagnostic_ids = mx.load(metrics.names[0] + "_ids.npy").tolist()
    print_residual_diagnostics("original weights", diagnostic_ids, km.weights)
    print_residual_diagnostics(f"{tag} weights", diagnostic_ids, qweights)
    for name in metrics.names:
        ids = mx.load(name + "_ids.npy")
        km.k_cache.clear()
        km.v_cache.clear()
        logits = km.forward(ids.tolist(), qweights, True, all_pos=True)
        mx.save(f"{name}_{tag}", logits)
        mx.save(f"{name}_{tag}_ids", ids)
    print(f"run: python metrics.py mine {tag}")
