from tokenizers import Tokenizer
import mlx.core as mx
import mlx.nn as nn
import config

tokenizer = Tokenizer.from_file("tokenizer.json")
weights = mx.load("model.safetensors")

rope = nn.RoPE(dims=config.head_dim, traditional=False, base=config.rope_theta)

text = "The cat sat"

def embed(input):
    output = tokenizer.encode(input)

    embed_matrix = weights["model.embed_tokens.weight"]
    output_vecotrs = embed_matrix[output.ids]

    return output_vecotrs

def rms_norm(x, weight, eps=config.rms_norm_eps):
    return mx.fast.rms_norm(x, weight, eps)

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

def attention(x, w, layer, prefill=False, pos=None):
    if prefill:
        p = f"model.layers.{layer}.self_attn."
        attn_heads = config.attention_heads
        kv_heads = config.kv_heads
        head_dim = config.head_dim


        q = x @ w[p + "q_proj.weight"].T + w[p + "q_proj.bias"]
        q = q.reshape(x.shape[0], attn_heads, head_dim)
        q = mx.transpose(q, axes=(1, 0, 2))
        q = rope(q)

        k = x @ w[p + "k_proj.weight"].T + w[p + "k_proj.bias"]
        k = k.reshape(x.shape[0], kv_heads, head_dim)
        k = mx.transpose(k, axes=(1, 0, 2))
        k = rope(k)
        kc = k

        v = x @ w[p + "v_proj.weight"].T + w[p + "v_proj.bias"]
        v = v.reshape(x.shape[0], kv_heads, head_dim)
        v = mx.transpose(v, axes=(1, 0, 2))
        vc = v

        out = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None],
            scale=head_dim**-0.5,
            mask="causal",
        )[0]
        length = q.shape[1]
        out = mx.transpose(out, axes=(1,0,2))
        out = out.reshape(length, config.hidden_size)
        out = out @ w[p + "o_proj.weight"].T

        return out, kc, vc
    p = f"model.layers.{layer}.self_attn."
    attn_heads = config.attention_heads
    kv_heads = config.kv_heads
    head_dim = config.head_dim

    q = x @ w[p + "q_proj.weight"].T + w[p + "q_proj.bias"]
    q = q.reshape(x.shape[0], attn_heads, head_dim)
    q = mx.transpose(q, axes=(1, 0, 2))
    # offset is only applied correctly with a leading batch axis
    q = rope(q[None], offset=pos)[0]

    k = x @ w[p + "k_proj.weight"].T + w[p + "k_proj.bias"]
    k = k.reshape(x.shape[0], kv_heads, head_dim)
    k = mx.transpose(k, axes=(1, 0, 2))
    k = rope(k[None], offset=pos)[0]
    k = mx.concatenate([k_cache[-1][layer], k], axis=1) 
    kc = k


    v = x @ w[p + "v_proj.weight"].T + w[p + "v_proj.bias"]
    v = v.reshape(x.shape[0], kv_heads, head_dim)
    v = mx.transpose(v, axes=(1, 0, 2))
    v = mx.concatenate([v_cache[-1][layer], v], axis=1)
    vc = v

    out = mx.fast.scaled_dot_product_attention(
        q[None], k[None], v[None],
        scale=head_dim**-0.5,
        mask=None,
    )[0]
    length = q.shape[1]
    out = mx.transpose(out, axes=(1,0,2))
    out = out.reshape(length, config.hidden_size)
    out = out @ w[p + "o_proj.weight"].T
    return out, kc, vc
k_cache = []
v_cache = []

def forward(ids, weights, prefill=False, pos=None, all_pos=False):
    if prefill:
        x = weights["model.embed_tokens.weight"][mx.array(ids)] 
        k_layers = []
        v_layers = []
        for layer in range(config.n_layers):
            normed = rms_norm(x, weights[f"model.layers.{layer}.input_layernorm.weight"])
            new, k, v = attention(normed, weights, layer, True)
            x = x + new
            normed = rms_norm(x, weights[f"model.layers.{layer}.post_attention_layernorm.weight"])
            x = x + mlp(normed, weights, layer)
            k_layers.append(k)
            v_layers.append(v)
        if all_pos:
            x = rms_norm(x, weights["model.norm.weight"])
        else:
            x = rms_norm(x[-1:], weights["model.norm.weight"])
        k_cache.append(k_layers)
        v_cache.append(v_layers)
        return x @ weights["model.embed_tokens.weight"].T    
    x = weights["model.embed_tokens.weight"][mx.array(ids)] 
    for layer in range(config.n_layers):
        normed = rms_norm(x, weights[f"model.layers.{layer}.input_layernorm.weight"])
        new, k, v = attention(normed, weights, layer, pos=pos)
        x = x + new
        normed = rms_norm(x, weights[f"model.layers.{layer}.post_attention_layernorm.weight"])
        x = x + mlp(normed, weights, layer)
        k_cache[-1][layer] = k
        v_cache[-1][layer] = v
    if all_pos:
        x = rms_norm(x, weights["model.norm.weight"])
    else:
        x = rms_norm(x[-1:], weights["model.norm.weight"])
    return x @ weights["model.embed_tokens.weight"].T       

def generate(text, n=50, temperature=0.2, top_p=0.8, penalty=1.1):
    ids = tokenizer.encode(text).ids
    logits = forward(ids, weights, True)[-1]        
    logits = repetition_penalty(logits, ids, penalty)
    next_id = sample(logits, temperature, top_p)
    ids.append(next_id)
    for _ in range(n-1):
        logits = forward([ids[-1]], weights, pos=len(ids)-1)[-1]        
        logits = repetition_penalty(logits, ids, penalty)
        next_id = sample(logits, temperature, top_p)
        ids.append(next_id)
    return tokenizer.decode(ids)

def save_logits(text, name, n=50, temperature=0.2, top_p=0.8, penalty=1.1):
    k_cache.clear()
    v_cache.clear()
    ids = tokenizer.encode(text).ids
    logits = forward(ids, weights, True, all_pos=True)
    mx.save(name, logits)
    mx.save(name + "_ids", mx.array(ids))
    return logits

lighthouse_prompt = config.lighthouse_prompt
python_prompt = config.python_prompt
reef_prompt = config.reef_prompt
soup_prompt = config.soup_prompt

if __name__ == "__main__":
    k_cache.clear()
    v_cache.clear()
    print(generate(lighthouse_prompt, n=100, temperature=0.2, top_p=0.8, penalty=1.1))