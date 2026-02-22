#!/usr/bin/env python3
"""
Prepare UniGraph FB15K237 processed files from:
https://github.com/villmow/datasets_knowledge_embedding

Outputs (under dataset/FB15K237/processed by default):
  - texts.pkl
  - geometric_data_processed.pt
  - data.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace as SN

# macOS OpenMP runtime conflict workaround (data prep script only).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


def find_first_existing(base: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Cannot parse triple line in {path}: {line}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def read_entity_texts(dataset_dir: Path) -> dict[str, str]:
    text_file = find_first_existing(dataset_dir, ["entity2textlong.txt", "entity2text.txt"])
    if text_file is not None:
        out: dict[str, str] = {}
        with text_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    ent, text = line.split("\t", 1)
                else:
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    ent, text = parts
                out[ent] = text.strip()
        return out

    wikidata_file = dataset_dir / "entity2wikidata.json"
    if wikidata_file.exists():
        out = {}
        data = json.loads(wikidata_file.read_text(encoding="utf-8"))
        for ent, meta in data.items():
            label = meta.get("label")
            desc = meta.get("description")
            if label and desc:
                out[ent] = f"{label}. {desc}"
            elif label:
                out[ent] = label
            elif desc:
                out[ent] = desc
            else:
                out[ent] = ent
        return out

    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to datasets_knowledge_embedding/FB15k-237",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset/FB15K237/processed",
        help="Output directory for UniGraph processed files",
    )
    args = parser.parse_args()

    src = Path(args.source_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = find_first_existing(src, ["train.txt", "train.tsv"])
    valid_path = find_first_existing(src, ["valid.txt", "valid.tsv"])
    test_path = find_first_existing(src, ["test.txt", "test.tsv"])
    if not train_path or not valid_path or not test_path:
        raise FileNotFoundError(
            "Missing split files. Expected train/valid/test as .txt or .tsv in source_dir."
        )

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test = read_triples(test_path)

    relations: dict[str, int] = {}
    entities: dict[str, int] = {}

    def get_rel_id(r: str) -> int:
        if r not in relations:
            relations[r] = len(relations)
        return relations[r]

    def get_ent_id(e: str) -> int:
        if e not in entities:
            entities[e] = len(entities)
        return entities[e]

    def encode_split(split: list[tuple[str, str, str]]) -> tuple[list[list[int]], list[int]]:
        pairs: list[list[int]] = []
        labels: list[int] = []
        for h, r, t in split:
            h_id = get_ent_id(h)
            t_id = get_ent_id(t)
            r_id = get_rel_id(r)
            pairs.append([h_id, t_id])
            labels.append(r_id)
        return pairs, labels

    train_pairs, train_labels = encode_split(train)
    valid_pairs, valid_labels = encode_split(valid)
    test_pairs, test_labels = encode_split(test)

    n_nodes = len(entities)
    all_pairs = train_pairs + valid_pairs + test_pairs
    edge_index = torch.tensor(all_pairs, dtype=torch.long).t().contiguous()

    # UniGraph only needs x shape and edge_index in load_kg_dataset.
    x = torch.arange(n_nodes, dtype=torch.long).view(-1, 1)
    y = torch.zeros(n_nodes, dtype=torch.long)
    graph_obj = SN(x=x, edge_index=edge_index, y=y)
    torch.save([graph_obj], out / "geometric_data_processed.pt")

    split_obj = {
        "train": (train_pairs, train_labels),
        "valid": (valid_pairs, valid_labels),
        "test": (test_pairs, test_labels),
    }
    torch.save([split_obj], out / "data.pt")

    text_map = read_entity_texts(src)
    id_to_ent = [None] * n_nodes
    for ent, idx in entities.items():
        id_to_ent[idx] = ent
    texts = [text_map.get(ent, ent) for ent in id_to_ent]
    torch.save([texts], out / "texts.pkl")

    print(f"Saved files to: {out}")
    print(f"Entities: {n_nodes}, Relations: {len(relations)}")
    print(f"Triples: train={len(train_pairs)} valid={len(valid_pairs)} test={len(test_pairs)}")


if __name__ == "__main__":
    main()
