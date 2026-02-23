"""
Inspect per-example predictions on the FB15K237 test set.

Usage:
    python inspect_predictions.py \
        --checkpoint checkpoints/pretrain_epoch_3.pt \
        --lm_type microsoft/deberta-base \
        --hidden_size 768 \
        --source_dir /path/to/FB15k-237   # optional: adds readable relation names
        --out predictions.csv

The script loads the model, gets embeddings, trains a linear probe (same as eval),
then saves a CSV with: head_text, tail_text, true_relation, pred_relation, correct.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TAG, TAGDataset
from model import UniGraph
from utils.functions import build_args, set_random_seed
from utils.evaluation import create_optimizer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_relation_names(source_dir: str) -> dict[int, str]:
    """Reconstruct relation-id → name mapping from original triple files.
    Must use same iteration order as prepare_fb15k237_from_villmow.py."""
    src = Path(source_dir)
    relations: dict[str, int] = {}

    def get_rel_id(r: str) -> int:
        if r not in relations:
            relations[r] = len(relations)
        return relations[r]

    def get_ent_id(e: str, entities: dict) -> int:
        if e not in entities:
            entities[e] = len(entities)
        return entities[e]

    entities: dict[str, int] = {}
    for split_file in ["train.txt", "valid.txt", "test.txt"]:
        path = src / split_file
        if not path.exists():
            path = src / split_file.replace(".txt", ".tsv")
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {split_file} in {source_dir}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    parts = line.strip().split()
                if len(parts) != 3:
                    continue
                h, r, t = parts
                get_ent_id(h, entities)
                get_ent_id(t, entities)
                get_rel_id(r)

    return {v: k for k, v in relations.items()}


class LogisticRegression(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_feat, num_classes)

    def forward(self, graph, x):
        return self.fc(x)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lm_type", type=str, default="microsoft/deberta-base")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--gnn_type", type=str, default="gat")
    parser.add_argument("--cut_off", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lp_epochs", type=int, default=100)
    parser.add_argument("--lr_f", type=float, default=1e-2)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Path to original FB15k-237 source dir for readable relation names")
    parser.add_argument("--out", type=str, default="predictions.csv")
    # passthrough args needed to satisfy build_args / TAG init
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--norm", type=str, default="layernorm")
    parser.add_argument("--negative_slope", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--process_mode", type=str, default="TA")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--run_entity", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--datasets_name", nargs="+", default=["FB15K237"])
    parser.add_argument("--eval_datasets_name", nargs="+", default=["FB15K237"])
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = UniGraph(args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # ── get embeddings ────────────────────────────────────────────────────────
    print("Computing embeddings...")
    eval_tag = TAG(args, "FB15K237")
    dataset = TAGDataset(eval_tag)
    loader = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size)

    output = []
    for batch, _ in tqdm(loader, desc="Embedding"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            emb = model.get_embeddings(batch)
        output.append(emb.cpu())
    output = torch.cat(output, 0)  # [n_entities, hidden]

    # ── build edge features ───────────────────────────────────────────────────
    node_pairs = torch.LongTensor(
        eval_tag.test_graph["train"][0] +
        eval_tag.test_graph["valid"][0] +
        eval_tag.test_graph["test"][0]
    )
    labels = torch.LongTensor(
        eval_tag.test_graph["train"][1] +
        eval_tag.test_graph["valid"][1] +
        eval_tag.test_graph["test"][1]
    )
    x = torch.cat([output[node_pairs[:, 0]], output[node_pairs[:, 1]]], dim=1)

    split_idx = eval_tag.split_idx
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    # ── train linear probe ────────────────────────────────────────────────────
    print("Training linear probe...")
    n_classes = eval_tag.data_info["n_labels"]
    encoder = LogisticRegression(x.shape[1], n_classes).to(device)
    optimizer = create_optimizer("adam", encoder, args.lr_f, args.weight_decay_f)
    criterion = nn.CrossEntropyLoss()

    x_dev = x.to(device)
    labels_dev = labels.to(device)
    graph = eval_tag.graph.to(device)

    best_val_acc = 0
    best_model = None

    for epoch in tqdm(range(args.lp_epochs), desc="Linear probe"):
        encoder.train()
        out = encoder(graph, x_dev)
        loss = criterion(out[train_idx], labels_dev[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder.eval()
            pred = encoder(graph, x_dev)
            val_acc = (pred[val_idx].argmax(1) == labels_dev[val_idx]).float().mean().item()
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(encoder)

    # ── get test predictions ──────────────────────────────────────────────────
    print("Getting test predictions...")
    best_model.eval()
    with torch.no_grad():
        logits = best_model(graph, x_dev)
        preds = logits.argmax(1).cpu()

    test_preds = preds[test_idx]
    test_labels = labels[test_idx]
    test_pairs = node_pairs[test_idx]

    # ── load entity texts ─────────────────────────────────────────────────────
    entity_texts = torch.load("dataset/FB15K237/processed/texts.pkl")[0]

    # ── load relation names (optional) ───────────────────────────────────────
    rel_names: dict[int, str] = {}
    if args.source_dir:
        print("Loading relation names from source dir...")
        rel_names = load_relation_names(args.source_dir)

    # ── write CSV ─────────────────────────────────────────────────────────────
    print(f"Writing {args.out}...")
    correct = (test_preds == test_labels).sum().item()
    total = len(test_labels)
    print(f"Test accuracy: {correct}/{total} = {correct/total:.4f}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head_text", "tail_text",
                         "true_relation_id", "true_relation_name",
                         "pred_relation_id", "pred_relation_name",
                         "correct"])
        for i in range(len(test_preds)):
            h_id = test_pairs[i, 0].item()
            t_id = test_pairs[i, 1].item()
            true_rel = test_labels[i].item()
            pred_rel = test_preds[i].item()
            writer.writerow([
                entity_texts[h_id],
                entity_texts[t_id],
                true_rel,
                rel_names.get(true_rel, ""),
                pred_rel,
                rel_names.get(pred_rel, ""),
                int(true_rel == pred_rel),
            ])

    print(f"Done. Saved to {args.out}")


if __name__ == "__main__":
    main()
