import mlx.core as mx
import kv_cache_model as km
import metrics

MATRIX = "model.layers.12.mlp.down_proj.weight"
LEVELS = [0.01, 0.1]

def noisy_weights(level, seed=0):
    w = km.weights[MATRIX].astype(mx.float32)
    mx.random.seed(seed)
    noise = mx.random.normal(w.shape) * (level * mx.std(w))
    weights = dict(km.weights)
    weights[MATRIX] = (w + noise).astype(km.weights[MATRIX].dtype)
    return weights

def logits_with(weights, ids):
    km.k_cache.clear()
    km.v_cache.clear()
    return km.forward(ids.tolist(), weights, True, all_pos=True).astype(mx.float32)

if __name__ == "__main__":
    print(f"matrix: {MATRIX}")
    for level in LEVELS:
        weights = noisy_weights(level)
        for name in metrics.names:
            ref = mx.load(name + ".npy").astype(mx.float32)
            ids = mx.load(name + "_ids.npy")
            mine = logits_with(weights, ids)
            kl_mean, kl_max = metrics.kl_divergence(ref, mine)
            top1 = metrics.top1_agreement(ref, mine)
            print(f"noise {level:>5.0%}  {name:28s} kl mean {kl_mean:.2e}  kl max {kl_max:.2e}  top1 {top1*100:6.2f}%")
