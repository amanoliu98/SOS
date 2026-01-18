import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.utils.prune as prune
from transformers import MambaPreTrainedModel, MambaModel, MambaConfig, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

class MambaClassificationHead(nn.Module):
    def __init__(self, config, custom_dropout=None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        dropout_prob = custom_dropout if custom_dropout is not None else 0.1
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class MambaForSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config, custom_dropout=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = MambaModel(config)
        self.score = MambaClassificationHead(config, custom_dropout)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, output_hidden_states=None, return_dict=None, **kwargs):
        outputs = self.backbone(input_ids, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = self.score(outputs[0][:, -1, :]) 
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

def parse_vector(vec_str):
    vals = [float(x) for x in vec_str.split(',')]
    return {
        'lr': vals[0],
        'dropout': vals[1],
        # SAFETY CLAMP
        'layers': max(6, int(round(vals[2]))), 
        'sparsity': vals[4],
        'quant_bits': int(round(vals[6]))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True); parser.add_argument("--vector", type=str, required=True)
    parser.add_argument("--gen", type=int, default=0); parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")
    
    # 20k samples, 7 epochs
    BATCH_SIZE, EPOCHS, MAX_LEN = 32, 7, 128 

    try:
        params = parse_vector(args.vector)
        print(f"   [Ind {args.id}] LR: {params['lr']:.7f} | Layers: {params['layers']} | Sparse: {params['sparsity']:.2f}", flush=True)
        
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        dataset = load_dataset("glue", "sst2")
        def tokenize_fn(x): return tokenizer(x["sentence"], padding="max_length", truncation=True, max_length=MAX_LEN)
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"]).rename_column("label", "labels")
        tokenized.set_format("torch")

        # 20,000 SAMPLES
        train_loader = DataLoader(tokenized["train"].shuffle().select(range(20000)), batch_size=BATCH_SIZE)
        val_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE)

        full_model = MambaForSequenceClassification.from_pretrained("state-spaces/mamba-130m-hf", num_labels=2, ignore_mismatched_sizes=True)
        config = full_model.config
        config.n_layer = params['layers']
        
        model = MambaForSequenceClassification(config, custom_dropout=params['dropout']).to(device)
        
        with torch.no_grad():
            model.backbone.embeddings.weight.copy_(full_model.backbone.embeddings.weight)
            for i in range(params['layers']):
                model.backbone.layers[i].load_state_dict(full_model.backbone.layers[i].state_dict())
        del full_model
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, len(train_loader)*EPOCHS)

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += loss.item()
            print(f"      Ind {args.id} | Ep {epoch+1} | Loss: {total_loss/len(train_loader):.4f}", flush=True)

        targets = []
        for l in model.backbone.layers:
            if hasattr(l.mixer, 'in_proj'): targets.append((l.mixer.in_proj, 'weight'))
        if targets:
            prune.global_unstructured(targets, prune.L1Unstructured, amount=params['sparsity'])
            for m, n in targets: prune.remove(m, n)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                correct += (torch.argmax(model(**batch).logits, -1) == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        val_acc = correct / total
        
        model.to("cpu")
        dummy = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN))
        start = time.time()
        for _ in range(50): _ = model(dummy)
        latency = ((time.time() - start) / 50) * 1000
        
        # GATEKEEPER
        speed_score = 1000.0 / (1000.0 + latency) 
        quant_bonus = (8 - params['quant_bits']) * 0.05
        
        if val_acc < 0.75:
            fitness = val_acc 
        else:
            fitness = (0.60 * val_acc) + (0.40 * speed_score) + quant_bonus
        
        print(f"RESULT|{args.id}|{val_acc:.4f}|{latency:.2f}|{fitness:.6f}", flush=True)

    except Exception as e:
        print(f"ERROR|{args.id}|{str(e)}", flush=True); print("0.0", flush=True)

if __name__ == "__main__": main()
