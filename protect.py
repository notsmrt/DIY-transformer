import sys
import mlx.core as mx
import kv_cache_model as km
import metrics
import config
from quantize_model import fake_quantize


BITS, GROUP = 4, 64

THRESHOLD = 100

def detect(w, threshold=THRESHOLD, top_k=3):
    flagged = {}
    print(f"{'layer':>5} | {'attn max|out|':>13} | {'mlp max|out|':>12} | where (prompt, pos, dim) | flag")
    for name in metrics.names:
        ids = mx.load(name + "_ids.npy").tolist()
        x = w["model.embed_tokens.weight"][mx.array(ids)]
        for layer in range(config.n_layers):
            normed = km.rms_norm(x, w[f"model.layers.{layer}.input_layernorm.weight"])
            attn, _, _ = km.attention(normed, w, layer, prefill=True)
            x = x + attn
            normed = km.rms_norm(x, w[f"model.layers.{layer}.post_attention_layernorm.weight"])
            mlp_out = km.mlp(normed, w, layer)
            x = x + mlp_out
            for block, out in (("attn", attn), ("mlp", mlp_out)):
                flat = mx.abs(out.astype(mx.float32)).reshape(-1)
                i = mx.argmax(flat).item()
                value = flat[i].item()
                key = (layer, block)
                if key not in flagged or value > flagged[key][0]:
                    flagged[key] = (value, name, i // out.shape[1], i % out.shape[1], normed)
    layers = []
    for layer in range(config.n_layers):
        a, m = flagged[(layer, "attn")], flagged[(layer, "mlp")]
        worst = m if m[0] >= a[0] else a
        flag = []
        if a[0] > threshold: flag.append("ATTN")
        if m[0] > threshold: flag.append("MLP"); layers.append(layer)
        short = worst[1].replace("logits_", "")
        print(f"{layer:5d} | {a[0]:13.1f} | {m[0]:12.1f} | {short:>14s} {worst[2]:4d} {worst[3]:4d} | {' '.join(flag)}")
    for layer in layers:
        _, name, pos, dim, normed = flagged[(layer, "mlp")]
        neurons(w, layer, pos, dim, normed, name, top_k)
    print(f"\nprotect these MLP layers: {' '.join(map(str, layers)) or '(none)'}")
    return layers

def neurons(w, layer, pos, dim, normed, name, top_k):
    p = f"model.layers.{layer}.mlp."
    gate = normed @ w[p + "gate_proj.weight"].T
    h = (gate * mx.sigmoid(gate)) * (normed @ w[p + "up_proj.weight"].T)
    hp = h[pos].astype(mx.float32)
    top = mx.argsort(-mx.abs(hp))[:top_k].tolist()
    down = w[p + "down_proj.weight"].astype(mx.float32)
    others = mx.concatenate([h[:pos], h[pos + 1:]])
    print(f"\nL{layer} MLP, {name} position {pos}, dim {dim}:")
    for n in top:
        print(f"  neuron {n:5d}: h = {hp[n].item():9.2f}   writes {(hp[n] * down[dim, n]).item():9.2f}"
              f"   (max |h| at other positions {mx.max(mx.abs(others[:, n])).item():.2f})")

def per_position_kl(ref, mine):
    p = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    q = mine - mx.logsumexp(mine, axis=-1, keepdims=True)
    return mx.sum(mx.exp(p) * (p - q), axis=-1)

def run(weights, ids):
    km.k_cache.clear(); km.v_cache.clear()
    return km.forward(ids.tolist(), weights, True, all_pos=True).astype(mx.float32)

def profile(kl, ids, bins=10, top=8):
    n = kl.shape[0]
    edges = [round(i * n / bins) for i in range(bins + 1)]
    print("  KL by tenth of the sequence (pos 0 excluded):")
    for i in range(bins):
        lo, hi = max(edges[i], 1), edges[i + 1]
        print(f"    [{lo:4d},{hi:4d})  mean {mx.mean(kl[lo:hi]).item():.4f}")
    order = mx.argsort(-kl)[:top].tolist()
    print("  worst positions (token predicted FROM):")
    for pos in order:
        ctx = km.tokenizer.decode(ids[max(0, pos - 6):pos + 1].tolist()).replace("\n", "\\n")
        print(f"    {pos:4d}  kl {kl[pos].item():7.3f}   ...{ctx!r}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] != "--detect":
        layers = [int(a) for a in args]
    else:
        layers = detect(km.weights)
        if args:
            sys.exit(0)
    if not layers:
        sys.exit(f"no MLP output exceeded {THRESHOLD}; pass layer indices manually")

    qall, _ = fake_quantize(km.weights, BITS, GROUP)
    def keep(layer_list, parts):
        w = dict(qall)
        for l in layer_list:
            for part in parts:
                key = f"model.layers.{l}.mlp.{part}.weight"
                w[key] = km.weights[key]
        return w
    MLP = ("gate_proj", "up_proj", "down_proj")
    base = f"q{BITS}g{GROUP}"
    variants = {"all q4": (qall, None)}
    for l in layers:
        variants[f"bf16 L{l} mlp"] = (keep([l], MLP), f"{base}_mlp{l}")
        variants[f"bf16 L{l} down"] = (keep([l], ("down_proj",)), f"{base}_down{l}")
    if len(layers) > 1:
        label, tag = "+".join(map(str, layers)), "-".join(map(str, layers))
        variants[f"bf16 L{label} mlp"] = (keep(layers, MLP), f"{base}_mlp{tag}")
        variants[f"bf16 L{label} down"] = (keep(layers, ("down_proj",)), f"{base}_down{tag}")

    print(f"\n{'prompt':28s} {'variant':16s} {'kl mean':>8s} {'kl pos0':>8s} {'kl mean 1:':>10s} {'top1':>7s}")
    kls = {}
    for name in metrics.names:
        ref = mx.load(name + ".npy").astype(mx.float32)
        ids = mx.load(name + "_ids.npy")
        for label, (w, tag) in variants.items():
            logits = run(w, ids)
            if tag:
                mx.save(f"{name}_{tag}", logits)
                mx.save(f"{name}_{tag}_ids", ids)
            kl = per_position_kl(ref, logits)
            mx.eval(kl)
            kls[(name, label)] = kl
            top1 = metrics.top1_agreement(ref, logits)
            print(f"{name:28s} {label:16s} {mx.mean(kl).item():8.4f} {kl[0].item():8.3f} "
                  f"{mx.mean(kl[1:]).item():10.4f} {top1*100:6.2f}%")

    for name in metrics.names:
        ids = mx.load(name + "_ids.npy")
        print(f"\n{name}  (all q4, {ids.shape[0]} tokens)")
        profile(kls[(name, 'all q4')], ids)

    print("\nsaved logits; score them with:")
    for _, tag in variants.values():
        if tag:
            print(f"  python metrics.py mine {tag}")
