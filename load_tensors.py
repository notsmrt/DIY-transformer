import mlx.core as mx

weights = mx.load("model.safetensors")

for key, value in weights.items():
    if ".0" in key and "attn" in key:
        print(key, value.shape, value.dtype)