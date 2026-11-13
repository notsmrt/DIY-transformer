from tokenizers import Tokenizer
import mlx.core as mx
import mlx.nn as nn
import config

tokenizer = Tokenizer.from_file("tokenizer.json")
weights = mx.load("model.safetensors")

text = "The cat sat"

def embed(input):
    output = tokenizer.encode(input)

    embed_matrix = weights["model.embed_tokens.weight"]
    output_vecotrs = embed_matrix[output.ids]

    return output_vecotrs

def rms_norm(x, weight, eps=config.rms_norm_eps):
    x32 = x.astype(mx.float32)                                  
    rms = mx.sqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return ((x32 / rms) * weight.astype(mx.float32)).astype(x.dtype)

def repetition_penalty(logits, ids, penalty=1.1):
    seen = mx.array(list(set(ids)))
    vals = logits[seen]
    logits[seen] = mx.where(vals > 0, vals / penalty, vals * penalty)
    return logits

def sample(logits, temperature=1.0, top_p=0.9):
    if temperature == 0:
        return mx.argmax(logits).item()
    logits = logits.astype(mx.float32) / temperature
    probs = mx.softmax(logits, axis=-1)

    order = mx.argsort(-probs)                      
    sorted_probs = probs[order]
    cumulative = mx.cumsum(sorted_probs)
    keep = (cumulative - sorted_probs) < top_p       
    sorted_probs = mx.where(keep, sorted_probs, 0.0)

    choice = mx.random.categorical(mx.log(sorted_probs))   
    return order[choice].item()                          

def mlp(x, w, layer):
    p = f"model.layers.{layer}.mlp."
    gate = x @ w[p + "gate_proj.weight"].T    
    up   = x @ w[p + "up_proj.weight"].T    
    gate = gate * mx.sigmoid(gate)          
    h = gate * up                       
    return h @ w[p + "down_proj.weight"].T   

def attention(x, w, layer):
    p = f"model.layers.{layer}.self_attn."
    attn_heads = config.attention_heads
    kv_heads = config.kv_heads
    head_dim = config.head_dim

    rope = nn.RoPE(dims=head_dim, traditional=False, base=config.rope_theta)

    q = x @ w[p + "q_proj.weight"].T + w[p + "q_proj.bias"]
    q = q.reshape(x.shape[0], attn_heads, head_dim)
    q = mx.transpose(q, axes=(1, 0, 2))
    q = rope(q)

    k = x @ w[p + "k_proj.weight"].T + w[p + "k_proj.bias"]
    k = k.reshape(x.shape[0], kv_heads, head_dim)
    k = mx.transpose(k, axes=(1, 0, 2))
    k = rope(k)
    k = mx.repeat(k, attn_heads//kv_heads, axis=0)
    k = mx.transpose(k, axes=(0, 2, 1))

    v = x @ w[p + "v_proj.weight"].T + w[p + "v_proj.bias"]
    v = v.reshape(x.shape[0], kv_heads, head_dim)
    v = mx.transpose(v, axes=(1, 0, 2))
    v = mx.repeat(v, attn_heads//kv_heads, axis=0)

    scores = q @ k / head_dim**0.5
    length = q.shape[1]
    mask = mx.full((length, length), -1e9)
    mask = mx.triu(mask, k=1)
    scores = scores + mask
    scores = mx.softmax(scores, axis=-1)

    out = scores @ v
    out = mx.transpose(out, axes=(1,0,2))
    out = out.reshape(length, config.hidden_size)
    out = out @ w[p + "o_proj.weight"].T
    
    return out

def forward(ids):
    x = weights["model.embed_tokens.weight"][mx.array(ids)] 
    for layer in range(config.n_layers):
        normed = rms_norm(x, weights[f"model.layers.{layer}.input_layernorm.weight"])
        x = x + attention(normed, weights, layer)
        normed = rms_norm(x, weights[f"model.layers.{layer}.post_attention_layernorm.weight"])
        x = x + mlp(normed, weights, layer)
    x = rms_norm(x[-1:], weights["model.norm.weight"])
    return x @ weights["model.embed_tokens.weight"].T       

def generate(text, n=500, temperature=0.5, top_p=0.9):
    ids = tokenizer.encode(text).ids
    for _ in range(n):
        logits = forward(ids)[-1]        
        logits = repetition_penalty(logits, ids, 1.1)
        next_id = sample(logits, temperature, top_p)
        ids.append(next_id)
    return tokenizer.decode(ids)

if __name__ == "__main__":
    print(generate(text))