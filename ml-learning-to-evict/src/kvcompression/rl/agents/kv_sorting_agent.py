#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from kvcompression.rl.agents.base_agent import BaseAgent


class KVSamplingGroupedQueryAgent(BaseAgent):
    """
    Agent class specifically designed for grouped query attention.
    Handles multiple queries per KV head from the start.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_length: int,
        processing_dim: int,
        mlp_hidden_size: int,
        pos_embedding_dim: int,
        num_mlp_layers: int,
        use_kv_queries: bool,
        device: Union[str, torch.device],
        query_aggregation: str = "mean",  # "mean", "max", "last", "learned"
        num_queries_per_group: Optional[
            int
        ] = None,  # Required for "learned" aggregation
        use_keydiff_features: bool = False,  # Whether to include KeyDiff similarity features
        keydiff_proj_dim: int = 16,  # Projection dimension for KeyDiff features
    ):
        """
        Initialize the KV sampling agent for grouped query attention.

        Args:
            head_dim: Dimension of each attention head.
            max_seq_length: Maximum sequence length for positional embeddings.
            processing_dim: Dimension for projected features in the MLP.
            mlp_hidden_size: Hidden layer size in the scoring MLP.
            pos_embedding_dim: Dimension of positional embeddings.
            num_mlp_layers: Number of hidden layers in the scoring MLP.
            use_kv_queries: Whether to include query features in scoring.
            device: Device to place the model on.
            query_aggregation: Method to aggregate queries per KV head:
                - "mean": Average across queries
                - "max": Max pooling across queries
                - "last": Use last query only
                - "learned": Use learned MLP aggregation
            num_queries_per_group: Number of queries per KV head. Required when
                query_aggregation="learned".
            use_keydiff_features: Whether to include KeyDiff similarity features.
            keydiff_proj_dim: Projection dimension for KeyDiff features.
        """
        super().__init__()
        self.use_kv_queries = use_kv_queries
        self.device = device
        self.max_seq_length = max_seq_length
        self.key_dim = head_dim
        self.value_dim = head_dim
        self.query_dim = head_dim
        self.processing_dim = processing_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.query_aggregation = query_aggregation
        self.num_queries_per_group = num_queries_per_group
        self.use_keydiff_features = use_keydiff_features
        self.keydiff_proj_dim = keydiff_proj_dim

        self.key_projection = nn.Linear(self.key_dim, processing_dim)
        self.value_projection = nn.Linear(self.value_dim, processing_dim)

        if self.use_kv_queries:
            self.query_projection = nn.Linear(self.query_dim, processing_dim)

        self.pos_embedding = nn.Embedding(max_seq_length, self.pos_embedding_dim)

        if self.use_keydiff_features:
            self.keydiff_projection = nn.Linear(1, keydiff_proj_dim)

        self.norm_key = nn.LayerNorm(processing_dim)
        self.norm_value = nn.LayerNorm(processing_dim)

        if self.use_kv_queries:
            self.norm_query = nn.LayerNorm(processing_dim)

        self.norm_pos = nn.LayerNorm(self.pos_embedding_dim)

        if self.use_keydiff_features:
            self.norm_keydiff = nn.LayerNorm(keydiff_proj_dim)

        if self.query_aggregation == "learned":
            if num_queries_per_group is None:
                raise ValueError(
                    "num_queries_per_group must be provided when using 'learned' query aggregation"
                )
            input_dim = num_queries_per_group * head_dim
            hidden_dim = head_dim
            self.query_aggregation_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, head_dim),
                nn.LayerNorm(head_dim),
            )
        else:
            self.query_aggregation_mlp = None

        # MLP input: concatenation of projected key, value, position, and optional query/keydiff features
        mlp_input_dim = processing_dim + processing_dim + self.pos_embedding_dim
        if self.use_kv_queries:
            mlp_input_dim += processing_dim
        if self.use_keydiff_features:
            mlp_input_dim += keydiff_proj_dim

        layers = []
        current_dim = mlp_input_dim
        for _ in range(num_mlp_layers):
            layers.append(nn.Linear(current_dim, mlp_hidden_size))
            layers.append(nn.LayerNorm(mlp_hidden_size))
            layers.append(nn.SiLU())
            current_dim = mlp_hidden_size
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.to(self.device)

    def requires_attention_weights(self) -> bool:
        return False

    def _compute_keydiff_features(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute KeyDiff similarity features using the same logic as KeyDiffPress.

        Args:
            keys: Key embeddings [B, S_padded, D_head]

        Returns:
            KeyDiff similarity scores [B, S_padded]
        """
        # Normalize keys and compute anchor (mean across sequence dimension)
        normalized_keys = F.normalize(keys, p=2, dim=-1)  # [B, S_padded, D_head]
        anchor = normalized_keys.mean(dim=1, keepdim=True)  # [B, 1, D_head]

        # Compute cosine similarity between keys and anchor
        # Using negative cosine similarity (lower score = higher similarity)
        keydiff_scores = -F.cosine_similarity(
            normalized_keys, anchor, dim=-1
        )  # [B, S_padded]

        return keydiff_scores

    def _aggregate_queries(self, queries, method="mean"):
        """
        Aggregate queries across the query group dimension.

        Args:
            queries: Query tensor of shape [B, queries_per_kv, S_padded, head_dim].
            method: Aggregation method ("mean", "max", "last", or "learned").

        Returns:
            Aggregated query tensor of shape [B, S_padded, head_dim].
        """
        if method == "mean":
            return queries.mean(dim=1)
        elif method == "max":
            return queries.max(dim=1)[0]
        elif method == "last":
            return queries[:, -1]
        elif method == "learned":
            batch_size, queries_per_kv, s_padded, head_dim = queries.shape

            if queries_per_kv != self.num_queries_per_group:
                raise RuntimeError(
                    f"Inconsistency between configured ({self.num_queries_per_group=}) and actual ({queries_per_kv=})"
                )

            # Reshape to process each position independently
            # [B, S_padded, queries_per_kv * D_head]
            queries_concat = queries.permute(0, 2, 1, 3)
            queries_concat = queries_concat.reshape(batch_size, s_padded, -1)
            aggregated = self.query_aggregation_mlp(queries_concat)
            return aggregated
        else:
            raise ValueError(f"Unknown query aggregation method: {method}")

    def forward(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute raw scores for each KV pair with grouped queries.

        Args:
            observation: Dictionary containing:
                - 'keys': Key embeddings [B, S_padded, D_head]
                - 'values': Value embeddings [B, S_padded, D_head]
                - 'queries': Query embeddings [B, queries_per_kv, S_padded, D_head]
                - 'seq_lengths': Valid sequence lengths [B]

        Returns:
            Dictionary containing:
                - 'scores': Raw score for selecting each item [B, S_padded]
        """
        keys = observation["keys"]  # [B, S_padded, D_head]
        values = observation["values"]  # [B, S_padded, D_head]
        item_queries = observation["queries"]  # [B, queries_per_kv, S_padded, D_head]
        seq_lengths = observation["seq_lengths"]

        if keys.ndim == 4:
            raise ValueError("Currently multi-KV head agent is not implemented")

        batch_size, s_padded, _ = keys.shape
        device = keys.device

        norm_keys = self.norm_key(self.key_projection(keys))
        norm_values = self.norm_value(self.value_projection(values))

        pos_indices = torch.arange(s_padded, device=device)
        positions_for_embedding = pos_indices.expand(batch_size, -1)
        norm_pos = self.norm_pos(
            self.pos_embedding(
                positions_for_embedding.clamp(max=self.max_seq_length - 1)
            )
        )

        if self.use_keydiff_features:
            keydiff_scores = self._compute_keydiff_features(keys)  # [B, S_padded]
            norm_keydiff = self.norm_keydiff(
                self.keydiff_projection(keydiff_scores.unsqueeze(-1))
            )  # [B, S_padded, keydiff_proj_dim]

        padding_mask = pos_indices.unsqueeze(0) < seq_lengths.unsqueeze(1)

        if self.use_kv_queries:
            # Aggregate queries across the query group dimension
            item_queries_agg = self._aggregate_queries(
                item_queries, self.query_aggregation
            )
            norm_item_queries = self.norm_query(self.query_projection(item_queries_agg))

            features_to_concat = [
                norm_keys,
                norm_values,
                norm_item_queries,
                norm_pos,
            ]
            if self.use_keydiff_features:
                features_to_concat.append(norm_keydiff)

            combined_features = torch.cat(features_to_concat, dim=-1)
        else:
            features_to_concat = [
                norm_keys,
                norm_values,
                norm_pos,
            ]
            if self.use_keydiff_features:
                features_to_concat.append(norm_keydiff)

            combined_features = torch.cat(features_to_concat, dim=-1)

        scores = self.mlp(combined_features).squeeze(-1)  # [B, S_padded]
        scores = scores.masked_fill(~padding_mask, -float("inf"))

        return {"scores": scores}

    def supports_gqa(self) -> bool:
        return True
