import sys
import mlx.core as mx
from mlx_lm import load
import config

REFERENCE = sys.argv[1] if len(sys.argv) > 1 else "mlx_lm"
_lm_model = None

names = [
    "logits_lighthouse",
    "logits_python_game",
    "logits_great_barrier_reef",
    "logits_vegetable_soup",
]

def reference_logits(ids, reference=None):
    reference = reference or REFERENCE
    if reference == "mine":
        import kv_cache_model as km
        km.k_cache.clear()
        km.v_cache.clear()
        return km.forward(ids.tolist(), km.weights, True, all_pos=True).astype(mx.float32)
    if reference == "mlx_lm":
        global _lm_model
        if _lm_model is None:
            _lm_model, _ = load(str(config.MODEL_DIR))
        return _lm_model(ids[None])[0].astype(mx.float32)
    raise ValueError(f"unknown reference {reference!r}, use 'mlx_lm' or 'mine'")

def kl_divergence(ref, mine):
    p_log = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    q_log = mine - mx.logsumexp(mine, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(p_log) * (p_log - q_log), axis=-1)
    return mx.mean(kl).item(), mx.max(kl).item()

def kl_positions(ref, mine, count=5):
    p_log = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    q_log = mine - mx.logsumexp(mine, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(p_log) * (p_log - q_log), axis=-1)
    top = mx.argsort(-kl)[:count]
    positions = top.tolist()
    values = kl[top].tolist()
    return mx.argmax(kl).item(), list(zip(positions, values))

def top1_agreement(ref, mine):
    return mx.mean(mx.argmax(ref, axis=-1) == mx.argmax(mine, axis=-1)).item()

def perplexity(logits, ids):
    log_probs = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    targets = mx.take_along_axis(log_probs, ids[1:, None], axis=-1)[:, 0]
    return mx.exp(-mx.mean(targets)).item()

if __name__ == "__main__":
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"reference: {REFERENCE}   logits: {tag or 'original'}")
    for name in names:
        file = f"{name}_{tag}" if tag else name
        mine = mx.load(file + ".npy").astype(mx.float32)
        ids = mx.load(file + "_ids.npy")
        ref = reference_logits(ids)

        kl_mean, kl_max = kl_divergence(ref, mine)
        max_position, top_positions = kl_positions(ref, mine)
        top1 = top1_agreement(ref, mine)
        ppl_mine = perplexity(mine, ids)
        ppl_ref = perplexity(ref, ids)

        print(name)
        print(f"  tokens:          {ids.shape[0]}")
        print(f"  kl mean:         {kl_mean:.6f}")
        print(f"  kl max:          {kl_max:.6f}")
        print(f"  max kl position: {max_position} (0-based)")
        print("  top 5 kl positions (0-based):")
        for position, kl_value in top_positions:
            print(f"    {position}: {kl_value:.6f}")
        print(f"  top1 agreement:  {top1 * 100:.2f}%")
        print(f"  perplexity mine: {ppl_mine:.4f}")
        print(f"  perplexity ref:  {ppl_ref:.4f}")
