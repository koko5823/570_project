
# -*- coding: utf-8 -*-
!pip -q install transformers datasets accelerate matplotlib pandas

import os
import random
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    set_seed as hf_set_seed,
)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TOKENIZER = "gpt2"
MAX_LENGTH = 128
TRAIN_SUBSET = 2000
VALID_SUBSET = 400

tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("wikitext", "wikitext-2-raw-v1")

train_text = raw["train"].select(range(TRAIN_SUBSET))
valid_text = raw["validation"].select(range(VALID_SUBSET))

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

train_tok = train_text.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_tok = valid_text.map(tokenize_fn, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("train features:", train_tok[0].keys())

class DyT(nn.Module):
    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

def make_small_gpt2(vocab_size: int):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=256,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(cfg)

def replace_layernorm_with_dyt(model: nn.Module, dim: int):
    def _replace(parent: nn.Module):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(parent, child_name, DyT(dim))
            else:
                _replace(child)
    _replace(model)
    return model

def build_model(use_dyt: bool):
    model = make_small_gpt2(len(tokenizer))
    if use_dyt:
        model = replace_layernorm_with_dyt(model, dim=256)
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def extract_train_losses(logs):
    xs, ys = [], []
    for item in logs:
        if "loss" in item and "step" in item and "eval_loss" not in item:
            xs.append(item["step"])
            ys.append(item["loss"])
    return xs, ys

def extract_eval_losses(logs):
    xs, ys = [], []
    for item in logs:
        if "eval_loss" in item and "epoch" in item:
            xs.append(item["epoch"])
            ys.append(item["eval_loss"])
    return xs, ys

def extract_epoch_train_losses(logs):
    """
    Optional summary: average train loss logged around each epoch.
    Since logging is step-based, this keeps the raw logged points with epoch tags.
    """
    rows = []
    for item in logs:
        if "loss" in item and "step" in item and "epoch" in item and "eval_loss" not in item:
            rows.append({
                "epoch": float(item["epoch"]),
                "step": int(item["step"]),
                "train_loss": float(item["loss"]),
            })
    return rows

def collect_dyt_alphas(model):
    alphas = {}
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, DyT):
            alphas[f"{idx}:{name}"] = float(module.alpha.detach().cpu().item())
            idx += 1
    return alphas


class AlphaLoggingTrainer(Trainer):
    def __init__(self, *args, track_alpha=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_alpha = track_alpha
        self.alpha_history = []

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.track_alpha:
            alpha_dict = collect_dyt_alphas(self.model)
            row = {"epoch": float(self.state.epoch) if self.state.epoch is not None else None}
            row.update(alpha_dict)
            self.alpha_history.append(row)
        return metrics

# =========================
# 6. Run one experiment
# =========================
def run_train_eval(seed: int, use_dyt: bool, out_dir: str, epochs: int = 5):
    set_all_seeds(seed)

    model = build_model(use_dyt=use_dyt).to(device)
    model_name = "DyT" if use_dyt else "LayerNorm"

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
    )

    trainer = AlphaLoggingTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
        track_alpha=use_dyt,
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    logs = trainer.state.log_history
    train_steps, train_losses = extract_train_losses(logs)
    eval_epochs, eval_losses = extract_eval_losses(logs)
    epoch_train_rows = extract_epoch_train_losses(logs)
    alpha_history = trainer.alpha_history if use_dyt else []

    run_summary = {
        "model": model_name,
        "seed": seed,
        "params": count_params(model),
        "final_eval_loss": float(final_metrics["eval_loss"]),
        "train_steps": train_steps,
        "train_losses": train_losses,
        "eval_epochs": eval_epochs,
        "eval_losses": eval_losses,
        "epoch_train_rows": epoch_train_rows,
        "alpha_history": alpha_history,
    }
    return run_summary

SEEDS = [42, 123, 456]
EPOCHS = 5
RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)

all_runs = []

for seed in SEEDS:
    print(f"\n=== Running LayerNorm, seed={seed} ===")
    ln_result = run_train_eval(
        seed=seed,
        use_dyt=False,
        out_dir=os.path.join(RESULT_DIR, f"ln_seed_{seed}"),
        epochs=EPOCHS,
    )
    all_runs.append(ln_result)

    print(f"\n=== Running DyT, seed={seed} ===")
    dyt_result = run_train_eval(
        seed=seed,
        use_dyt=True,
        out_dir=os.path.join(RESULT_DIR, f"dyt_seed_{seed}"),
        epochs=EPOCHS,
    )
    all_runs.append(dyt_result)

with open(os.path.join(RESULT_DIR, "all_runs.json"), "w") as f:
    json.dump(all_runs, f, indent=2)

summary_rows = []
epoch_rows = []
alpha_rows = []

for run in all_runs:
    summary_rows.append({
        "model": run["model"],
        "seed": run["seed"],
        "params": run["params"],
        "final_eval_loss": run["final_eval_loss"],
    })

    for ep, ev in zip(run["eval_epochs"], run["eval_losses"]):
        epoch_rows.append({
            "model": run["model"],
            "seed": run["seed"],
            "epoch": ep,
            "eval_loss": ev,
        })

    for row in run["epoch_train_rows"]:
        epoch_rows.append({
            "model": run["model"],
            "seed": run["seed"],
            "epoch": row["epoch"],
            "step": row["step"],
            "train_loss": row["train_loss"],
        })

    for row in run["alpha_history"]:
        row_copy = {"model": run["model"], "seed": run["seed"]}
        row_copy.update(row)
        alpha_rows.append(row_copy)

summary_df = pd.DataFrame(summary_rows)
epoch_df = pd.DataFrame(epoch_rows)
alpha_df = pd.DataFrame(alpha_rows) if alpha_rows else pd.DataFrame()

summary_df.to_csv(os.path.join(RESULT_DIR, "summary.csv"), index=False)
epoch_df.to_csv(os.path.join(RESULT_DIR, "epoch_logs.csv"), index=False)
if not alpha_df.empty:
    alpha_df.to_csv(os.path.join(RESULT_DIR, "alpha_history.csv"), index=False)

print("\n=== Per-run final eval loss ===")
print(summary_df)

agg_df = (
    summary_df.groupby("model")["final_eval_loss"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
agg_df.to_csv(os.path.join(RESULT_DIR, "aggregate_summary.csv"), index=False)

print("\n=== Aggregate summary (mean ± std) ===")
print(agg_df)

def average_eval_curve(all_runs, model_name):
    rows = []
    for run in all_runs:
        if run["model"] != model_name:
            continue
        for ep, loss in zip(run["eval_epochs"], run["eval_losses"]):
            rows.append({"epoch": ep, "eval_loss": loss})
    df = pd.DataFrame(rows)
    grouped = df.groupby("epoch")["eval_loss"].agg(["mean", "std"]).reset_index()
    return grouped

ln_eval_avg = average_eval_curve(all_runs, "LayerNorm")
dyt_eval_avg = average_eval_curve(all_runs, "DyT")

plt.figure(figsize=(7, 5))
plt.plot(ln_eval_avg["epoch"], ln_eval_avg["mean"], label="LayerNorm eval loss")
plt.plot(dyt_eval_avg["epoch"], dyt_eval_avg["mean"], label="DyT eval loss")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation Loss over 5 Epochs on WikiText-2 Subset")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "eval_loss_curve.png"), dpi=200)
plt.show()

def get_run(all_runs, model_name, seed):
    for run in all_runs:
        if run["model"] == model_name and run["seed"] == seed:
            return run
    return None

plot_seed = SEEDS[0]
ln_run = get_run(all_runs, "LayerNorm", plot_seed)
dyt_run = get_run(all_runs, "DyT", plot_seed)

plt.figure(figsize=(7, 5))
plt.plot(ln_run["train_steps"], ln_run["train_losses"], label=f"LayerNorm train loss (seed={plot_seed})")
plt.plot(dyt_run["train_steps"], dyt_run["train_losses"], label=f"DyT train loss (seed={plot_seed})")
plt.xlabel("Step")
plt.ylabel("Training loss")
plt.title("Training Loss over 5 Epochs on WikiText-2 Subset")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "train_loss_curve_seed42.png"), dpi=200)
plt.show()

if not alpha_df.empty:
    alpha_cols = [c for c in alpha_df.columns if c not in ["model", "seed", "epoch"]]

    # only plot first 2 alpha traces from first seed to keep it clean
    first_seed_alpha = alpha_df[alpha_df["seed"] == SEEDS[0]].sort_values("epoch")

    if len(alpha_cols) > 0:
        plt.figure(figsize=(7, 5))
        for col in alpha_cols[:2]:
            plt.plot(first_seed_alpha["epoch"], first_seed_alpha[col], marker="o", label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Learned alpha")
        plt.title(f"DyT Alpha over Epochs (seed={SEEDS[0]})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "dyt_alpha_curve_seed42.png"), dpi=200)
        plt.show()

print(f"\nAll results saved in: {RESULT_DIR}")