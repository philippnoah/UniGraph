import os
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, get_scheduler, AutoTokenizer
from tqdm.auto import tqdm
import wandb
import dgl
import argparse
import numpy as np

from data import *
from model import UniGraph
from utils.functions import create_optimizer, get_current_lr, set_random_seed, drop_edge, pool, Evaluator
from utils.evaluation import node_classification_evaluation, link_prediction_evaluation, edge_classification_evaluation, graph_classification_evaluation, incontext_evaluate
from utils.data_util import preprocess
from gensim.models import KeyedVectors
from ppr import PPRSampler
from instruction_tuning import GraphInstructionTuning

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--lm_type", type=str, default="microsoft/deberta-base")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--norm", type=str, default="layernorm")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--delayed_ema_epoch", type=int, default=10)
    
    # PPR sampling parameters
    parser.add_argument("--ppr_alpha", type=float, default=0.15)
    parser.add_argument("--ppr_top_k", type=int, default=32)
    
    # Instruction tuning parameters
    parser.add_argument("--llm_type", type=str, default="meta-llama/Llama-2-7b")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    return parser.parse_args()

def evaluate(args, model, device, name=""):
    for dataset in args.eval_datasets_name:
        if dataset in ['cora', 'pubmed', 'arxiv', 'products', 'wikics', 'FB15K237', 'WN18RR']:
            evaluate_tag(args, model, device, dataset)
        elif args.task in ["w2cnc", "w2cec"]:
            evaluate_w2v(args, model, device, dataset)
        else:
            evaluate_mol(args, model, device, dataset)
    return

def evaluate_tag(args, model, device, name=""):
    eval_tag = TAG(args, name)
    dataset = TAGDataset(eval_tag)
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size)
    model.eval()
    output = []

    for batch, _ in tqdm(eval_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        emb = model.emb(batch)
        output.append(emb.cpu())
    output = torch.cat(output, 0)

    if args.incontext_eval:
        acc = incontext_evaluate(args, output, name)
        graph = eval_tag.graph
        graph.ndata["feat"] = output
        output, cat_output = model.inference(graph, device, args.eval_batch_size)
        acc = incontext_evaluate(args, output, name)
        acc = incontext_evaluate(args, cat_output, name)

    if name in ['cora', 'pubmed', 'arxiv', 'products', 'wikics']:
        graph = eval_tag.graph
        test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(graph, output, eval_tag.labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
        wandb.log({
            f"{name}_estp_test_acc_lm": estp_test_acc,
            f"{name}_best_val_acc_lm": best_val_acc,
        })

        if args.gnn_type != "":
            graph.ndata["feat"] = output
            output, cat_output = model.inference(graph, device, args.eval_batch_size)
            test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(graph, output, eval_tag.labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
            wandb.log({
                f"{name}_estp_test_acc_gnn": estp_test_acc,
                f"{name}_best_val_acc_gnn": best_val_acc,
            })

    elif name in ['FB15K237', 'WN18RR']:
        graph = eval_tag.graph
        node_pairs = torch.LongTensor(eval_tag.test_graph["train"][0] + eval_tag.test_graph["valid"][0] + eval_tag.test_graph["test"][0])
        labels = torch.LongTensor(eval_tag.test_graph["train"][1] + eval_tag.test_graph["valid"][1] + eval_tag.test_graph["test"][1])
        test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
        wandb.log({
            f"{name}_estp_test_acc_lm": estp_test_acc,
            f"{name}_best_val_acc_lm": best_val_acc,
        })
        if args.gnn_type != "":
            graph.ndata["feat"] = output
            output, cat_output = model.inference(graph, device, args.eval_batch_size)
            test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
            wandb.log({
                f"{name}_estp_test_acc_gnn": estp_test_acc,
                f"{name}_best_val_acc_gnn": best_val_acc,
            })
            test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, cat_output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
            wandb.log({
                f"{name}_estp_test_acc_gnn_cat": estp_test_acc,
                f"{name}_best_val_acc_gnn_cat": best_val_acc,
            })
    else:
        graph = eval_tag.test_graph
        link_prediction_evaluation(graph, output)

    return output

def evaluate_mol(args, model, device, name=""):
    eval_mol = Mol(args, name)
    dataset = IterMolDataset(eval_mol, 0, args.batch_size)
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=None)
    model.eval()
    pooler = pool(args.pooler)
    output_lm, output_gnn, output_gnn_cat = [], [], []
    labels = []

    if args.incontext_eval:
        acc = incontext_evaluate(args, None, output_lm, name)
        acc = incontext_evaluate(args, None, output_gnn, name)
        acc = incontext_evaluate(args, None, output_gnn_cat, name)

    evaluator = Evaluator(name='ogbg-molhiv' if name == "hiv" else 'ogbg-molpcba' if name == "pcba" else 'ogbg-molchembl')
    test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(output_lm, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks, args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device)
    wandb.log({
        "estp_test_acc_lm": estp_test_acc,
        "best_val_acc_lm": best_val_acc,
    })
    if args.gnn_type != "":
        test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(output_gnn, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks, args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device)
        if best_val_acc > evaluate_mol.g_val_acc_gnn:
            evaluate_mol.g_val_acc_gnn = best_val_acc
            evaluate_mol.g_test_acc_gnn = estp_test_acc
        wandb.log({
            "estp_test_acc_gnn": estp_test_acc,
            "best_val_acc_gnn": best_val_acc,
        })

def train_pretrain(args, model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_latent_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Get graph embeddings
        graph_emb = model.get_embeddings(batch)
        
        # Sample subgraphs using PPR
        sampler = PPRSampler(alpha=args.ppr_alpha, top_k=args.ppr_top_k)
        sampled_nodes, sampled_edges = sampler.sample_subgraph(batch["graph"], graph_emb)
        
        # Forward pass
        loss, latent_loss = model(batch, sampled_nodes, epoch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_latent_loss += latent_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "latent_loss": f"{latent_loss.item():.4f}"
        })
    
    return total_loss / len(train_loader), total_latent_loss / len(train_loader)

def train_instruction_tuning(args, model, instruction_tuner, train_loader, optimizer, epoch):
    model.eval()
    instruction_tuner.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Instruction Tuning Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Get graph embeddings
        with torch.no_grad():
            graph_emb = model.get_embeddings(batch)
        
        # Instruction tuning step
        loss, metrics = instruction_tuner.train_step(
            graph_emb,
            batch["instruction"],
            batch["target"],
            batch["task_type"],
            batch.get("label_space")
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "perplexity": f"{metrics['perplexity']:.4f}"
        })
    
    return total_loss / len(train_loader)

def main():
    args = parse_args()
    set_random_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lm_type)
    mask_token_id = tokenizer.mask_token_id
    
    # Load dataset
    train_dataset = TextAttributedGraphDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    model = UniGraph(args).to(args.device)
    
    # Initialize instruction tuner
    instruction_tuner = GraphInstructionTuning(
        base_model_name=args.llm_type,
        graph_embedding_dim=args.hidden_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ).to(args.device)
    
    # Initialize optimizers
    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    instruction_optimizer = torch.optim.AdamW(
        instruction_tuner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    for epoch in range(args.epochs):
        # Pre-training phase
        pretrain_loss, pretrain_latent_loss = train_pretrain(
            args, model, train_loader, pretrain_optimizer, epoch
        )
        
        # Instruction tuning phase
        instruction_loss = train_instruction_tuning(
            args, model, instruction_tuner, train_loader, instruction_optimizer, epoch
        )
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": pretrain_optimizer.state_dict(),
                "pretrain_loss": pretrain_loss,
                "pretrain_latent_loss": pretrain_latent_loss
            }, os.path.join(args.save_dir, f"pretrain_epoch_{epoch+1}.pt"))
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": instruction_tuner.state_dict(),
                "optimizer_state_dict": instruction_optimizer.state_dict(),
                "instruction_loss": instruction_loss
            }, os.path.join(args.save_dir, f"instruction_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()
