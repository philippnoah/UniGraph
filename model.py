import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

import dgl
from transformers import AutoModel

from gat import GAT
from utils.loss_func import sce_loss, compute_mlm_loss
from utils.functions import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class UniGraph(nn.Module):
    """UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs"""
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Language model encoder
        self.lm_encoder = self._load_lm_encoder(args.lm_type)

        # GNN encoder: takes LM [CLS] embeddings as node features
        self.gnn_encoder = GAT(
            in_dim=args.hidden_size,
            num_hidden=args.hidden_size,
            out_dim=args.hidden_size,
            num_layers=args.num_layers,
            nhead=args.nhead,
            nhead_out=args.nhead,
            activation=args.activation,
            feat_drop=args.dropout,
            attn_drop=args.dropout,
            negative_slope=args.negative_slope,
            residual=True,
            norm=args.norm,
            concat_out=True,
            encoding=True,
        )

        # Fusion layer to combine LM and GNN outputs
        self.fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU()
        )

        # Projector and target networks for latent space regularization (BGRL-style)
        if args.lam > 0:
            self.projector = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
            )

            # Target networks are updated via EMA and process unmasked inputs
            self.target_lm_encoder = self._load_lm_encoder(args.lm_type)
            self.target_gnn_encoder = GAT(
                in_dim=args.hidden_size,
                num_hidden=args.hidden_size,
                out_dim=args.hidden_size,
                num_layers=args.num_layers,
                nhead=args.nhead,
                nhead_out=args.nhead,
                activation=args.activation,
                feat_drop=0.0,   # no dropout in target network
                attn_drop=0.0,
                negative_slope=args.negative_slope,
                residual=True,
                norm=args.norm,
                concat_out=True,
                encoding=True,
            )
            self.target_fusion = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
                nn.ReLU()
            )
            self.target_projector = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
            )

            # Initialize target networks as copies of online networks
            self._init_target_networks()

        # MLM head: predicts masked tokens from full sequence LM hidden states
        self.mlm_head = nn.Linear(
            self.lm_encoder.config.hidden_size,
            self.lm_encoder.config.vocab_size
        )

    def _init_target_networks(self):
        """Initialize target networks with the same weights as online networks."""
        self.target_lm_encoder.load_state_dict(self.lm_encoder.state_dict())
        self.target_gnn_encoder.load_state_dict(self.gnn_encoder.state_dict())
        self.target_fusion.load_state_dict(self.fusion.state_dict())
        self.target_projector.load_state_dict(self.projector.state_dict())

        # Freeze target networks (updated only via EMA)
        for param in self.target_lm_encoder.parameters():
            param.requires_grad = False
        for param in self.target_gnn_encoder.parameters():
            param.requires_grad = False
        for param in self.target_fusion.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @staticmethod
    def _load_lm_encoder(model_name: str):
        return AutoModel.from_pretrained(model_name, use_safetensors=True)

    @torch.no_grad()
    def _update_target_networks(self, momentum=0.99):
        """Update target networks using exponential moving average."""
        for target_param, param in zip(self.target_lm_encoder.parameters(), self.lm_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data
        for target_param, param in zip(self.target_gnn_encoder.parameters(), self.gnn_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data
        for target_param, param in zip(self.target_fusion.parameters(), self.fusion.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data
        for target_param, param in zip(self.target_projector.parameters(), self.projector.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing MLM loss and optional latent space regularization loss.

        Args:
            batch: dict with keys:
                - masked_input_ids: token ids with random tokens replaced by [MASK]
                - input_ids: original (unmasked) token ids — used as MLM labels
                - attention_mask: attention mask
                - token_type_ids (optional): token type ids
                - graph: DGL graph of the current subgraph

        Returns:
            (total_loss, latent_loss)
        """
        masked_input_ids = batch["masked_input_ids"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch.get("token_type_ids")
        graph = batch["graph"]

        # Ensure graph has self-loops (required by GATConv)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        # --- Online network: processes MASKED text ---
        lm_kwargs = dict(input_ids=masked_input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            lm_kwargs["token_type_ids"] = token_type_ids
        lm_outputs = self.lm_encoder(**lm_kwargs)

        # Node feature = [CLS] token representation
        node_features = lm_outputs.last_hidden_state[:, 0]  # [batch, hidden]

        # GNN encoding over the subgraph
        graph_embeddings = self.gnn_encoder(graph, node_features)  # [batch, hidden]

        # Fuse LM and GNN representations
        combined = self.fusion(torch.cat([node_features, graph_embeddings], dim=-1))

        # MLM loss: predict masked tokens using full sequence hidden states
        mlm_logits = self.mlm_head(lm_outputs.last_hidden_state)  # [batch, seq_len, vocab]
        mlm_loss = compute_mlm_loss(mlm_logits, input_ids, masked_input_ids)

        # --- Latent space regularization (BGRL-style) ---
        latent_loss = torch.tensor(0.0, device=mlm_loss.device)

        if self.args.lam > 0:
            with torch.no_grad():
                # Target network processes UNMASKED inputs for stable targets
                target_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
                if token_type_ids is not None:
                    target_kwargs["token_type_ids"] = token_type_ids
                target_lm_out = self.target_lm_encoder(**target_kwargs)
                target_node_features = target_lm_out.last_hidden_state[:, 0]
                target_graph_emb = self.target_gnn_encoder(graph, target_node_features)
                target_combined = self.target_fusion(
                    torch.cat([target_node_features, target_graph_emb], dim=-1)
                )
                target_embeddings = self.target_projector(target_combined)

            online_embeddings = self.projector(combined)
            latent_loss = sce_loss(online_embeddings, target_embeddings)

            # EMA update of target networks
            self._update_target_networks(self.args.momentum)

        total_loss = mlm_loss + self.args.lam * latent_loss
        return total_loss, latent_loss

    def get_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Get LM [CLS] embeddings for each node (no GNN).
        Used during evaluation to collect per-node representations before
        optionally running full-graph GNN inference via self.inference().
        """
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch.get("token_type_ids")

            lm_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                lm_kwargs["token_type_ids"] = token_type_ids
            lm_outputs = self.lm_encoder(**lm_kwargs)
            node_features = lm_outputs.last_hidden_state[:, 0]
            return node_features

    def inference(self, graph, device, batch_size: int = 128):
        """
        Run layer-wise GNN inference on the full graph.
        Expects graph.ndata["feat"] to contain node features (e.g. LM embeddings).

        Returns:
            (output, cat_output): final GNN embeddings and concatenation of all layer outputs
        """
        return self.gnn_encoder.inference(graph, device, batch_size)
