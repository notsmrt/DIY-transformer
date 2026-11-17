import mlx.core as mx
from mlx_lm import load

model, tokenizer = load(".")

names = [
    "logits_lighthouse",
    "logits_python_game",
    "logits_great_barrier_reef",
    "logits_vegetable_soup",
]

def reference_logits(ids):
    return model(ids[None])[0].astype(mx.float32)

def kl_divergence(ref, mine):
    p_log = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    q_log = mine - mx.logsumexp(mine, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(p_log) * (p_log - q_log), axis=-1)
    return mx.mean(kl).item(), mx.max(kl).item()

def top1_agreement(ref, mine):
    return mx.mean(mx.argmax(ref, axis=-1) == mx.argmax(mine, axis=-1)).item()

def perplexity(logits, ids):
    log_probs = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    targets = mx.take_along_axis(log_probs, ids[1:, None], axis=-1)[:, 0]
    return mx.exp(-mx.mean(targets)).item()

if __name__ == "__main__":
    for name in names:
        mine = mx.load(name + ".npy").astype(mx.float32)
        ids = mx.load(name + "_ids.npy")
        ref = reference_logits(ids)

        kl_mean, kl_max = kl_divergence(ref, mine)
        top1 = top1_agreement(ref, mine)
        ppl_mine = perplexity(mine, ids)
        ppl_ref = perplexity(ref, ids)

        print(name)
        print(f"  tokens:          {ids.shape[0]}")
        print(f"  kl mean:         {kl_mean:.6f}")
        print(f"  kl max:          {kl_max:.6f}")
        print(f"  top1 agreement:  {top1 * 100:.2f}%")
        print(f"  perplexity mine: {ppl_mine:.4f}")
        print(f"  perplexity ref:  {ppl_ref:.4f}")
