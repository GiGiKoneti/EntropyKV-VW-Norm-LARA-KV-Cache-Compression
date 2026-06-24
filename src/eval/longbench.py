import os
import argparse
import time
import re
import string
import collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.baselines.streaming import StreamingCache
from src.baselines.h2o import H2OCache
from src.baselines.snapkv import SnapKVCache
from src.baselines.random_eviction import RandomCache
from src.cache.entropy_cache import EntropyCache

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Standard SQuAD/QA evaluation metrics (F1 and EM)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
        
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1

def run_downstream_qa(model, tokenizer, document, question, reference, cache_class, cache_kwargs, device="cpu"):
    """
    Prompts the model with a long document and a question, then generates and evaluates the response.
    """
    # Format the prompt using the model's chat template if available
    try:
        messages = [
            {"role": "user", "content": f"Document:\n{document}\n\nQuestion: {question}\nAnswer: the answer is"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to default raw string
        prompt = f"Document:\n{document}\n\nQuestion: {question}\nAnswer: the answer is"
        
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    cache = None
    if cache_class is not None:
        cache = cache_class(**cache_kwargs)
        
    output_attentions = cache_class is not None and hasattr(cache_class, "accumulate_attentions")
    
    # Prefill Phase
    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=output_attentions
        )
    cache = outputs.past_key_values
        
    if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
        cache.accumulate_attentions(outputs.attentions)
        
    if hasattr(cache, "evict"):
        cache.evict()
        
    # Generate 20 tokens
    curr_pos = input_ids.shape[1]
    next_input = outputs.logits[:, -1:, :].argmax(dim=-1)
    generated_tokens = [next_input.item()]
    print(f"[DEBUG] Prompt length: {curr_pos}, first token ID: {next_input.item()}, decoded: '{tokenizer.decode([next_input.item()])}'")
    
    for step in range(19):
        pos_ids = torch.tensor([[curr_pos]], device=device)
        with torch.no_grad():
            outputs = model(
                next_input,
                past_key_values=cache,
                use_cache=True,
                position_ids=pos_ids,
                output_attentions=output_attentions
            )
        cache = outputs.past_key_values
            
        if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
            cache.accumulate_attentions(outputs.attentions)
            
        if hasattr(cache, "evict"):
            cache.evict()
            
        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated_tokens.append(next_tok.item())
        print(f"[DEBUG] Step {step+1}: predicted token ID: {next_tok.item()}, decoded: '{tokenizer.decode([next_tok.item()])}'")
        next_input = next_tok
        curr_pos += 1
        
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"[DEBUG] Final decoded prediction: '{prediction}'")
    
    em = compute_exact_match(prediction, reference)
    f1 = compute_f1(prediction, reference)
    
    return prediction, em, f1

def main():
    parser = argparse.ArgumentParser(description="Lightweight Long-Context Downstream Task Evaluator")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--budget_ratio", type=float, default=0.5, help="KV cache budget ratio")
    parser.add_argument("--method", type=str, default="entropykv", choices=["full", "streaming", "h2o", "snapkv", "random", "entropykv"])
    parser.add_argument("--metric", type=str, default="l2_norm", choices=["l2_norm", "variance", "shannon", "vw_norm"])
    parser.add_argument("--layer_adaptive", action="store_true", help="Use layer-adaptive budgets (U-shape curve)")
    parser.add_argument("--vw_gamma", type=float, default=0.0, help="Exponent for Value vector in Value-Weighted Key Norm")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of evaluation samples")
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recency_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    if "qwen" in args.model.lower() and device == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=device,
        attn_implementation="eager"
    )
    model.eval()
    
    # Try to load a standard long-context dataset. Fallback to constructed Wikitext document if offline/error.
    eval_samples = []
    
    try:
        print("Attempting to load 'allenai/qasper' dataset from HuggingFace...")
        # Load validation set
        ds = load_dataset("allenai/qasper", split="train") # Qasper has a single train split with standard subsets
        # We extract up to num_samples papers and QA pairs
        count = 0
        for paper in ds:
            paper_text = ""
            for sec in paper["full_text"]["paragraphs"]:
                paper_text += "\n".join(sec) + "\n\n"
            
            # Keep document under ~3000 tokens for local efficiency
            if len(tokenizer.tokenize(paper_text)) > 3000:
                paper_text = paper_text[:12000] # truncate
                
            qas = paper["qas"]
            for qa in qas:
                question = qa["question"]
                # Get the first free-text or yes-no answer
                answers = qa["answers"]
                ans_text = None
                for ans in answers:
                    if ans["answer"]["free_text"] != "":
                        ans_text = ans["answer"]["free_text"]
                        break
                    elif ans["answer"]["yes_no"] is not None:
                        ans_text = "yes" if ans["answer"]["yes_no"] else "no"
                        break
                
                if ans_text and len(paper_text) > 2000:
                    eval_samples.append({
                        "document": paper_text,
                        "question": question,
                        "answer": ans_text
                    })
                    count += 1
                    break # one QA per paper
            if count >= args.num_samples:
                break
    except Exception as e:
        print(f"Failed to load 'allenai/qasper': {e}")
        print("Falling back to constructing synthetic long-context QA samples from WikiText-2...")
        
        # Robust synthetic fallback
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        full_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip() != ""])
        
        # We split full text into blocks of 4000 characters (approx 1000 tokens)
        for i in range(args.num_samples):
            start_pos = i * 4000
            doc_chunk = full_text[start_pos : start_pos + 4000]
            
            # Let's insert a synthetic QA pair in the document chunk
            magic_number = args.seed + i * 17
            needle_sentence = f" The crucial scientific result reveals that gravity is {magic_number} times weaker on this planet."
            
            # Insert needle at 40% of the chunk
            insert_idx = int(0.4 * len(doc_chunk))
            doc_chunk_with_needle = doc_chunk[:insert_idx] + needle_sentence + doc_chunk[insert_idx:]
            
            eval_samples.append({
                "document": doc_chunk_with_needle,
                "question": "How many times weaker is gravity on this planet according to the result?",
                "answer": f"{magic_number} times"
            })
            
    print(f"Prepared {len(eval_samples)} downstream QA samples for evaluation.")
    
    # Configure pluggable Cache
    cache_class = None
    cache_kwargs = {}
    
    # We will use an average document length of ~2000 tokens for setup
    avg_doc_len = 2048
    max_cache_size = int(avg_doc_len * args.budget_ratio)
    
    if args.method == "full":
        cache_class = None
        cache_kwargs = {}
    elif args.method == "streaming":
        cache_class = StreamingCache
        cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size}
    elif args.method == "h2o":
        cache_class = H2OCache
        cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size, "recency_size": args.recency_size}
    elif args.method == "snapkv":
        cache_class = SnapKVCache
        cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size, "recency_size": args.recency_size}
    elif args.method == "random":
        cache_class = RandomCache
        cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size, "recency_size": args.recency_size}
    elif args.method == "entropykv":
        cache_class = EntropyCache
        cache_kwargs = {
            "max_cache_size": max_cache_size,
            "metric": args.metric,
            "sink_size": args.sink_size,
            "recency_size": args.recency_size,
            "layer_adaptive": args.layer_adaptive,
            "vw_gamma": args.vw_gamma
        }
        
    total_em = 0.0
    total_f1 = 0.0
    
    start_time = time.time()
    
    for idx, sample in enumerate(eval_samples):
        print(f"\nEvaluating Sample {idx+1}/{len(eval_samples)}...")
        # Re-initialize budget-ratio cache based on exact doc token length for this run
        doc_tokens = tokenizer(sample["document"], return_tensors="pt").input_ids
        current_doc_len = doc_tokens.size(1)
        max_cache_size = int(current_doc_len * args.budget_ratio)
        
        if cache_class is not None:
            cache_kwargs["max_cache_size"] = max_cache_size
            
        pred, em, f1 = run_downstream_qa(
            model=model,
            tokenizer=tokenizer,
            document=sample["document"],
            question=sample["question"],
            reference=sample["answer"],
            cache_class=cache_class,
            cache_kwargs=cache_kwargs,
            device=device
        )
        
        total_em += em
        total_f1 += f1
        print(f"Question:  {sample['question']}")
        print(f"Reference: {sample['answer']}")
        print(f"Predicted: {pred}")
        print(f"EM: {em} | F1: {f1:.4f}")
        
    avg_em = total_em / len(eval_samples)
    avg_f1 = total_f1 / len(eval_samples)
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("                 DOWNSTREAM QA RESULTS")
    print("="*50)
    print(f"Model:          {args.model}")
    print(f"Method:         {args.method.upper()}")
    if args.method == "entropykv":
        print(f"Metric:         {args.metric.upper()}")
    print(f"Budget Ratio:   {args.budget_ratio:.2f}")
    print(f"Total Samples:  {len(eval_samples)}")
    print("-"*50)
    print(f"Average EM:     {avg_em:.4f}")
    print(f"Average F1:     {avg_f1:.4f}")
    print(f"Elapsed Time:   {elapsed:.2f} seconds")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
