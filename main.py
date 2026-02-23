import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import dgl
import numpy as np

from data import *
from model import UniGraph
from utils.functions import (
    build_args, create_optimizer, get_current_lr, set_random_seed, drop_edge, pool, Evaluator
)
from utils.evaluation import (
    node_classification_evaluation, edge_classification_evaluation,
    graph_classification_evaluation, incontext_evaluate
)
from utils.data_util import preprocess
from instruction_tuning import GraphInstructionTuning


def evaluate(args, model, device, name=""):
    for dataset in args.eval_datasets_name:
        if dataset in ['cora', 'pubmed', 'arxiv', 'products', 'wikics', 'FB15K237', 'WN18RR']:
            evaluate_tag(args, model, device, dataset)
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
        with torch.no_grad():
            emb = model.get_embeddings(batch)
        output.append(emb.cpu())
    output = torch.cat(output, 0)

    if args.incontext_eval:
        incontext_evaluate(args, output, name)
        graph = eval_tag.graph
        graph.ndata["feat"] = output
        output, cat_output = model.inference(graph, device, args.eval_batch_size)
        incontext_evaluate(args, output, name)
        incontext_evaluate(args, cat_output, name)

    if name in ['cora', 'pubmed', 'arxiv', 'products', 'wikics']:
        graph = eval_tag.graph
        test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(
            graph, output, eval_tag.labels, eval_tag.split_idx,
            eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f,
            args.lp_epochs, device
        )
        wandb.log({
            f"{name}_estp_test_acc_lm": estp_test_acc,
            f"{name}_best_val_acc_lm": best_val_acc,
        })

        if args.gnn_type != "":
            graph.ndata["feat"] = output
            with torch.no_grad():
                output, cat_output = model.inference(graph, device, args.eval_batch_size)
            test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(
                graph, output, eval_tag.labels, eval_tag.split_idx,
                eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f,
                args.lp_epochs, device
            )
            wandb.log({
                f"{name}_estp_test_acc_gnn": estp_test_acc,
                f"{name}_best_val_acc_gnn": best_val_acc,
            })

    elif name in ['FB15K237', 'WN18RR']:
        graph = eval_tag.graph
        node_pairs = torch.LongTensor(
            eval_tag.test_graph["train"][0] + eval_tag.test_graph["valid"][0] + eval_tag.test_graph["test"][0]
        )
        labels = torch.LongTensor(
            eval_tag.test_graph["train"][1] + eval_tag.test_graph["valid"][1] + eval_tag.test_graph["test"][1]
        )
        test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(
            graph, output, node_pairs, labels, eval_tag.split_idx,
            eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f,
            args.lp_epochs, device
        )
        wandb.log({
            f"{name}_estp_test_acc_lm": estp_test_acc,
            f"{name}_best_val_acc_lm": best_val_acc,
        })
        if args.gnn_type != "":
            graph.ndata["feat"] = output
            with torch.no_grad():
                output, cat_output = model.inference(graph, device, args.eval_batch_size)
            test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(
                graph, output, node_pairs, labels, eval_tag.split_idx,
                eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f,
                args.lp_epochs, device
            )
            wandb.log({
                f"{name}_estp_test_acc_gnn": estp_test_acc,
                f"{name}_best_val_acc_gnn": best_val_acc,
            })
            test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(
                graph, cat_output, node_pairs, labels, eval_tag.split_idx,
                eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f,
                args.lp_epochs, device
            )
            wandb.log({
                f"{name}_estp_test_acc_gnn_cat": estp_test_acc,
                f"{name}_best_val_acc_gnn_cat": best_val_acc,
            })

    return output


def evaluate_mol(args, model, device, name=""):
    eval_mol = Mol(args, name)
    dataset = IterMolDataset(eval_mol, 0, args.batch_size)
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=None)
    model.eval()
    pooler = pool(args.pooler)
    output_lm = []
    g_val_acc_gnn = 0.0
    g_test_acc_gnn = 0.0

    for batch, iter_data, idx in tqdm(eval_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            emb = model.get_embeddings(batch)
            graph_emb = pooler(iter_data.to(device), emb)
        output_lm.append(graph_emb.cpu())

    output_lm = torch.cat(output_lm, 0)

    if args.incontext_eval:
        incontext_evaluate(args, output_lm, name)

    evaluator = Evaluator(
        name='ogbg-molhiv' if name == "hiv" else 'ogbg-molpcba' if name == "pcba" else 'ogbg-molchembl'
    )
    test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(
        output_lm, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks,
        args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device
    )
    wandb.log({
        f"{name}_estp_test_acc_lm": estp_test_acc,
        f"{name}_best_val_acc_lm": best_val_acc,
    })


def train_pretrain(args, model, train_loader, tag_datasets, optimizer, epoch, device, global_step, eval_fn=None, save_dir=None):
    model.train()
    total_loss = 0
    total_latent_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch, iter_data, dataset_idx in pbar:
        optimizer.zero_grad()

        # Move token tensors to device; graph is handled separately
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        subgraph = dgl.node_subgraph(tag_datasets[dataset_idx].graph, iter_data).to(device)
        batch["graph"] = subgraph

        # Forward pass (graph is expected in batch["graph"] when using TextAttributedGraphDataset,
        # or must be constructed from iter_data + full graph for IterTAGDataset)
        loss, latent_loss = model(batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_latent_loss += latent_loss.item()

        wandb.log({
            "global_step": global_step,
            "train_loss_step": loss.item(),
            "train_latent_loss_step": latent_loss.item(),
            "epoch": epoch,
        }, step=global_step)
        global_step += 1

        if eval_fn is not None and args.eval_interval > 0 and global_step % args.eval_interval == 0:
            model.eval()
            eval_fn()
            model.train()

        if save_dir is not None and args.save_interval > 0 and global_step % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pretrain_loss": loss.item(),
            }, os.path.join(save_dir, f"pretrain_step_{global_step}.pt"))

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "latent_loss": f"{latent_loss.item():.4f}"
        })

    return total_loss / len(train_loader), total_latent_loss / len(train_loader), global_step


def train_instruction_tuning(args, model, instruction_tuner, train_loader, optimizer, epoch, device):
    model.eval()
    instruction_tuner.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Instruction Tuning Epoch {epoch}")
    for batch, iter_data, dataset_idx in pbar:
        optimizer.zero_grad()

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            graph_emb = model.get_embeddings(batch)

        loss, metrics = instruction_tuner.train_step(
            graph_emb,
            batch["instruction"],
            batch["target"],
            batch["task_type"],
            batch.get("label_space")
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "perplexity": f"{metrics['perplexity']:.4f}"
        })

    return total_loss / len(train_loader)


def main():
    args = build_args()
    set_random_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    save_dir = args.checkpoint_path if args.checkpoint_path else "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Load training dataset(s)
    tag_datasets = []
    iter_datasets = []
    for name in args.datasets_name:
        tag = TAG(args, name)
        tag_datasets.append(tag)
        iter_datasets.append(
            IterTAGDataset(tag, len(iter_datasets), args.batch_size, args.num_roots, args.length)
        )

    train_dataset = CombinedDataset(iter_datasets, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

    # Initialize model
    model = UniGraph(args).to(device)

    if args.load_checkpoint and args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logging.info(f"Loaded checkpoint from {args.checkpoint_path}")

    pretrain_optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    wandb.init(project="unigraph", name=args.run_name, entity=args.run_entity, config=vars(args))

    # Pre-training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        eval_fn = (lambda: evaluate(args, model, device)) if args.eval_interval > 0 else None
        pretrain_loss, pretrain_latent_loss, global_step = train_pretrain(
            args, model, train_loader, tag_datasets, pretrain_optimizer, epoch, device, global_step, eval_fn=eval_fn, save_dir=save_dir
        )
        logging.info(
            f"Epoch {epoch}: loss={pretrain_loss:.4f}, latent_loss={pretrain_latent_loss:.4f}"
        )
        wandb.log({
            "epoch": epoch,
            "pretrain_loss": pretrain_loss,
            "pretrain_latent_loss": pretrain_latent_loss,
        })

        if (epoch + 1) % args.eval_steps == 0 or epoch == args.num_epochs - 1:
            evaluate(args, model, device)

        if (epoch + 1) % args.save_steps == 0 or epoch == args.num_epochs - 1:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": pretrain_optimizer.state_dict(),
                "pretrain_loss": pretrain_loss,
            }, os.path.join(save_dir, f"pretrain_epoch_{epoch + 1}.pt"))


if __name__ == "__main__":
    main()
