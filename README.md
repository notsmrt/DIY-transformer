# DIY transformer

I built this to teach myself how a transformer actually works on the inside. Instead of calling a library that hides everything behind `model.generate()`, I load the raw weights of a small pretrained model and write every step of the forward pass by hand: embedding lookup, RMSNorm, rotary position embeddings, grouped-query attention, the causal mask, the SwiGLU MLP, the residual connections, and sampling.

The model is Qwen2.5-0.5B (24 layers, hidden size 896, 14 attention heads sharing 2 key/value heads). It is small enough to run on a laptop and it uses the same building blocks as much larger models. [MLX](https://github.com/ml-explore/mlx) handles the array math and nothing else. The only `mlx.nn` piece I use is the RoPE helper.

The code favors being readable over being fast. There is no KV cache, so every new token reruns the whole sequence through all 24 layers.

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

### `bench.py`: measure speed

```sh
python bench.py
```

This feeds the model a prompt of about 700 tokens, generates 100 more, and prints:

- `ttft`: time to the first token, in seconds
- `itls`: the time each later token took
- `average itl`: the mean time per token
- `average token per second`: generation speed

It runs one untimed warm-up pass first, so the numbers do not include one-time startup cost.

### `load_tensors.py`: inspect the weights

```sh
python load_tensors.py
```

This prints the name, shape, and data type of every attention weight in layer 0. It is useful for seeing how the tensors in the weights file are named and how big they are. Edit the filter in the loop to look at other layers or other parts of the model.

## Files

| File | What it is |
| --- | --- |
| `model.py` | The forward pass, sampling, and generation loop |
| `config.py` | The model's dimensions, copied from `config.json` |
| `bench.py` | The speed benchmark |
| `token_ids.py` | A tokenizer and embedding demo |
| `load_tensors.py` | A weight inspector |
| `config.json` | The model's original config from Hugging Face, kept for reference |
| `model.safetensors`, `tokenizer.json` | The downloaded weights and tokenizer |
