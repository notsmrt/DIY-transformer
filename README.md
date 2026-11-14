# DIY transformer

I built this to teach myself how a transformer actually works on the inside. Instead of calling a library that hides everything behind `model.generate()`, I load the raw weights of a small pretrained model and write every step of the forward pass by hand: embedding lookup, RMSNorm, rotary position embeddings, grouped-query attention, the causal mask, the SwiGLU MLP, the residual connections, and sampling.

The model is Qwen2.5-0.5B (24 layers, hidden size 896, 14 attention heads sharing 2 key/value heads). It is small enough to run on a laptop and it uses the same building blocks as much larger models. [MLX](https://github.com/ml-explore/mlx) handles the array math.

There are two versions of the model:

- `model.py` is the reference implementation. It favors being readable over being fast: every step is written out, the only MLX helper it uses is `nn.RoPE`, and there is no KV cache, so every new token reruns the whole sequence through all 24 layers.
- `kv_cache_model.py` is the fast version. It adds a KV cache and swaps the hand-written RMSNorm and attention math for MLX's `mx.fast` kernels. Both versions produce the same output for the same prompt when sampling is greedy.

## Setup

You need a Mac with Apple Silicon (MLX requires it) and Python 3.9 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install mlx tokenizers huggingface_hub
```

The weights and tokenizer are not part of the code, so download them into the project folder:

```sh
hf download Qwen/Qwen2.5-0.5B model.safetensors tokenizer.json config.json --local-dir .
```

The scripts load `model.safetensors` and `tokenizer.json` by relative path, so run them from the project folder.

## Scripts

### `model.py`: generate text

```sh
python model.py
```

This continues the prompt `"The cat sat"` for 500 tokens and prints the result. To use a different prompt, change the `text` variable near the top of the file.

This is a base model, not a chat model, so it continues whatever text you give it and does not answer questions.

You can also call it from your own code:

```python
from model import generate

print(generate("Once upon a time", n=100, temperature=0.5, top_p=0.9))
```

| Argument | Default | What it does |
| --- | --- | --- |
| `n` | `500` | Number of tokens to generate |
| `temperature` | `0.5` | Higher is more random. `0` always picks the most likely token |
| `top_p` | `0.9` | Samples only from the most likely tokens that add up to this probability |

A repetition penalty of 1.1 is always applied to tokens that have already appeared.

### `kv_cache_model.py`: generate text, faster

```sh
python kv_cache_model.py
```

This continues the prompt `"The cat sat"` for 50 tokens. It is the same model as `model.py` with two changes that make it faster.

**A KV cache.** Generation is split into two phases:

1. Prefill: the whole prompt goes through all 24 layers once, and each layer saves the keys and values it computed.
2. Decode: each later step sends only the newest token through the model. Its query attends over the saved keys and values plus its own, and its key and value are appended to the cache for the next step.

A token's keys and values never change once they are computed, so recomputing them for every new token, as `model.py` does, is wasted work. With the cache, each step runs one token through the layers instead of the whole sequence.

Because the new token is processed on its own, RoPE has to be told where it sits in the sequence. The decode step passes the token's position as the `offset`.

**MLX's `mx.fast` kernels.** Two pieces of hand-written math are replaced with MLX's optimized versions:

- `mx.fast.rms_norm` replaces the manual RMSNorm.
- `mx.fast.scaled_dot_product_attention` replaces the manual scores, mask, softmax, and weighted sum. It builds the causal mask itself (`mask="causal"`) and broadcasts the 2 key/value heads across the 14 query heads, so the keys and values are cached as 2 heads and never repeated to 14.

Calling it from your own code works the same way as `model.py`:

```python
from kv_cache_model import generate

print(generate("Once upon a time", n=100, temperature=0.5, top_p=0.9))
```

| Argument | Default | What it does |
| --- | --- | --- |
| `n` | `50` | Number of tokens to generate |
| `temperature` | `0.2` | Higher is more random. `0` always picks the most likely token |
| `top_p` | `0.8` | Samples only from the most likely tokens that add up to this probability |
| `penalty` | `1.1` | Repetition penalty applied to tokens that have already appeared |

### `bench.py` and `bench_kv_cache.py`: measure speed

```sh
python bench.py            # model.py, no cache
python bench_kv_cache.py   # kv_cache_model.py
```

Both feed the model the same prompt of about 700 tokens, generate 100 more with the same sampling settings, and print:

- `ttft`: time to the first token, in seconds. For the cached model this is the prefill
- `itls`: the time each later token took
- `average itl`: the mean time per token
- `average token per second`: generation speed

Each runs one untimed warm-up pass first, so the numbers do not include one-time startup cost. Run both to see what the cache and the `mx.fast` kernels buy you.

### `bench_mlx.py`: compare against `mlx-lm`

```sh
pip install mlx-lm
python bench_mlx.py
```

This runs the same benchmark through [`mlx-lm`](https://github.com/ml-explore/mlx-lm), MLX's own generation library, loading the weights from the project folder. It prints the same four numbers, which gives a baseline to compare the hand-written versions against. It needs `config.json` in the project folder.

### `load_tensors.py`: inspect the weights

```sh
python load_tensors.py
```

This prints the name, shape, and data type of every attention weight in layer 0. It is useful for seeing how the tensors in the weights file are named and how big they are. Edit the filter in the loop to look at other layers or other parts of the model.

## Files

| File | What it is |
| --- | --- |
| `model.py` | The reference forward pass, sampling, and generation loop |
| `kv_cache_model.py` | The same model with a KV cache and `mx.fast` kernels |
| `config.py` | The model's dimensions, copied from `config.json` |
| `bench.py` | The speed benchmark for `model.py` |
| `bench_kv_cache.py` | The speed benchmark for `kv_cache_model.py` |
| `bench_mlx.py` | The same benchmark run through `mlx-lm` |
| `load_tensors.py` | A weight inspector |
| `config.json` | The model's original config from Hugging Face |
| `model.safetensors`, `tokenizer.json` | The downloaded weights and tokenizer |
