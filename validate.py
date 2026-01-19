import torch
import time
import numpy as np
import gc
import copy
import os
import torch.nn as nn
from transformers import MambaConfig, MambaModel, MambaPreTrainedModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# --- DEFINITIONS ---
class MambaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)
    def forward(self, features):
        x = self.dense(features)
        x = torch.tanh(x)
        return self.out_proj(x)

class MambaForSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)
        self.score = MambaClassificationHead(config)
    def forward(self, input_ids):
        outputs = self.backbone(input_ids)
        return self.score(outputs[0][:, -1, :]) # Last token

def get_dataloader():
    print("  Loading SST-2 Validation Set...")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("glue", "sst2", split="validation")
    
    # Tokenize
    def tokenize_fn(x): return tokenizer(x["sentence"], padding="max_length", truncation=True, max_length=128)
    tokenized = dataset.map(tokenize_fn, batched=True)
    
    # Rename 'label' to 'labels' if it exists
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    
    # Keep only necessary columns
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    cols_to_remove = [c for c in tokenized.column_names if c not in cols_to_keep]
    tokenized = tokenized.remove_columns(cols_to_remove)
    
    tokenized.set_format("torch")
    return DataLoader(tokenized, batch_size=1) 

def evaluate_model(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    
    # 1. PARAMETERS
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # 2. ACCURACY CHECK
    print("  Checking Accuracy...")
    correct = 0
    total = 0
    limit = 200 # Check 200 samples
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= limit: break
            input_ids = batch['input_ids'].to(device)
            # Handle case where labels might be missing or named differently
            if 'labels' in batch:
                labels = batch['labels'].to(device)
                logits = model(input_ids)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += 1
            else:
                print("  Warning: No labels found in batch!")
                break
    
    acc = correct / (total + 1e-8)
    print(f"  Validation Accuracy (Subset {total}): {acc:.4%}")

    # 3. SPEED TEST
    print("  Benchmarking Latency...")
    dummy_input = torch.randint(0, 1000, (1, 128)).to(device)
    
    for _ in range(10): _ = model(dummy_input) # Warmup
    
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(dummy_input)
            latencies.append((time.time() - start) * 1000)
    
    avg_lat = np.mean(latencies)
    print(f"  Avg Latency: {avg_lat:.2f} ms")
    
    return total_params, avg_lat, acc

def main():
    print("==============================================")
    print("STARTING FINAL VALIDATION (Size + Speed + Acc)")
    print("==============================================")
    
    dataloader = get_dataloader()
    
    # --- 1. BASELINE ---
    print("\n--- MEASURING BASELINE (Mamba-130m) ---")
    base_model = MambaForSequenceClassification.from_pretrained("state-spaces/mamba-130m-hf", num_labels=2, ignore_mismatched_sizes=True)
    p_base, t_base, acc_base = evaluate_model(base_model, dataloader)
    del base_model
    gc.collect()

    # --- 2. CHAMPION ---
    print("\n--- MEASURING CHAMPION (8-Layer) ---")
    
    # 1. Load FULL Pre-trained Model to get weights
    print("  Loading weights from pretrained...")
    full_model = MambaForSequenceClassification.from_pretrained("state-spaces/mamba-130m-hf", num_labels=2, ignore_mismatched_sizes=True)
    
    # 2. Create target config
    config = copy.deepcopy(full_model.config)
    config.n_layer = 8
    config.num_hidden_layers = 8 # Fix for HF config aliasing
    
    # 3. Create SLICED model
    champ_model = MambaForSequenceClassification(config)
    
    # 4. Apply Weights
    if os.path.exists("mamba_gold_model.pt"):
        print("  Loading Fine-Tuned 'mamba_gold_model.pt'...")
        # Load state dict, but handle potential key mismatches if saved differently
        state_dict = torch.load("mamba_gold_model.pt", map_location="cpu")
        champ_model.load_state_dict(state_dict)
    else:
        print("  [WARNING] 'mamba_gold_model.pt' NOT FOUND! Slicing raw weights instead.")
        with torch.no_grad():
            champ_model.backbone.embeddings.weight.copy_(full_model.backbone.embeddings.weight)
            for i in range(8):
                champ_model.backbone.layers[i].load_state_dict(full_model.backbone.layers[i].state_dict())
    
    del full_model
    gc.collect()
    
    p_champ, t_champ, acc_champ = evaluate_model(champ_model, dataloader)

    print("\n=== FINAL REPORT CARD ===")
    print(f"{'Metric':<15} | {'Baseline':<15} | {'Champion':<15} | {'Improvement'}")
    print("-" * 65)
    print(f"{'Parameters':<15} | {p_base/1e6:.1f}M            | {p_champ/1e6:.1f}M            | {(1 - p_champ/p_base)*100:.1f}% Smaller")
    print(f"{'Latency':<15} | {t_base:.2f} ms        | {t_champ:.2f} ms        | {t_base/t_champ:.2f}x Faster")
    print(f"{'Accuracy':<15} | {acc_base:.2%} (Raw)     | {acc_champ:.2%}         | Validated")

if __name__ == "__main__":
    main()
