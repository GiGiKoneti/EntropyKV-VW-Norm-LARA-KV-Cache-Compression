import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEQ_LEN = 1024
NUM_SAMPLES = 5
OUTPUT_DIR = "analysis/extracted_data"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Determine Device
    # Force 'cpu' to avoid MPS backend out of memory errors due to large eager attention matrices
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")
    
    # 2. Load Tokenizer and Model
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # TinyLlama has GQA. We load in float16/bfloat16 if device supports it, otherwise float32.
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype=model_dtype,
        device_map=device,
        attn_implementation="eager"  # Required for output_attentions=True
    )
    model.eval()
    
    # Get GQA parameters
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_queries_per_kv = num_heads // num_kv_heads
    print(f"Model config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}")
    
    # 3. Load and Prepare Dataset (WikiText-2 validation split)
    print("Loading WikiText-2 validation split...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Tokenize and concatenate text
    print("Preparing dataset text and chunking...")
    full_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip() != ""])
    tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
    
    num_tokens = len(tokens)
    print(f"Total tokens in validation split: {num_tokens}")
    
    # Chunk into sequences of length SEQ_LEN
    chunks = []
    for start in range(0, num_tokens - SEQ_LEN, SEQ_LEN):
        chunks.append(tokens[start : start + SEQ_LEN])
        if len(chunks) >= NUM_SAMPLES:
            break
            
    print(f"Prepared {len(chunks)} chunks of length {SEQ_LEN}.")
    
    # 4. Process Chunks
    for idx, chunk in enumerate(chunks):
        out_path = os.path.join(OUTPUT_DIR, f"sample_{idx}.npz")
        if os.path.exists(out_path):
            print(f"\nSample {idx} already exists at {out_path}. Skipping.")
            continue
            
        print(f"\nProcessing chunk {idx + 1}/{NUM_SAMPLES}...")
        input_ids = chunk.unsqueeze(0).to(device) # Shape: (1, SEQ_LEN)
        
        # Run forward pass, collecting attentions
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, use_cache=True)
            
        past_key_values = outputs.past_key_values
        attentions = outputs.attentions
        
        # DynamicCache exposes .layers in transformers >= 5.0
        num_layers = len(past_key_values.layers)
        
        # Save keys and cumulative attention weights
        # We aggregate attention weights for GQA to map them to KV heads
        layer_keys = []
        layer_cum_att = []
        
        for layer_idx in range(num_layers):
            # Key tensor shape: (batch_size, num_kv_heads, seq_len, head_dim)
            # We squeeze batch dimension (1)
            key_tensor = past_key_values.layers[layer_idx].keys.squeeze(0).cpu().to(torch.float32).numpy()
            layer_keys.append(key_tensor)
            
            # Attention tensor shape: (batch_size, num_heads, seq_len, seq_len)
            # Squeeze batch dimension (1)
            att_tensor = attentions[layer_idx].squeeze(0) # Shape: (num_heads, seq_len, seq_len)
            
            # Map query attention heads to KV heads by reshaping and summing
            # shape: (num_kv_heads, num_queries_per_kv, seq_len, seq_len)
            att_gqa = att_tensor.view(num_kv_heads, num_queries_per_kv, SEQ_LEN, SEQ_LEN)
            # Sum over query heads associated with each KV head
            att_gqa = att_gqa.sum(dim=1) # Shape: (num_kv_heads, seq_len, seq_len)
            
            # Cumulative attention is the column sum (sum over all query positions for each key position)
            # Shape: (num_kv_heads, seq_len)
            cum_att = att_gqa.sum(dim=1).cpu().to(torch.float32).numpy()
            layer_cum_att.append(cum_att)
            
        # Convert list of arrays to single numpy arrays
        # Shape: (num_layers, num_kv_heads, seq_len, head_dim)
        keys_np = np.stack(layer_keys, axis=0)
        # Shape: (num_layers, num_kv_heads, seq_len)
        cum_att_np = np.stack(layer_cum_att, axis=0)
        
        # Save npz file
        out_path = os.path.join(OUTPUT_DIR, f"sample_{idx}.npz")
        np.savez_compressed(
            out_path,
            keys=keys_np,
            cum_att=cum_att_np
        )
        print(f"Saved extracted data to {out_path}")
        print(f"  keys shape: {keys_np.shape}")
        print(f"  cum_att shape: {cum_att_np.shape}")

if __name__ == "__main__":
    main()
