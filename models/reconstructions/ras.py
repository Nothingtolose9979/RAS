import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

def add_jitter(feature_tokens, scale=20.0, prob=1.0):
    if random.uniform(0, 1) <= prob:
        num_tokens, batch_size, dim_channel = feature_tokens.shape
        feature_norms = (
            feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
        )  # (H x W) x B x 1
        jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
        jitter = jitter * feature_norms * scale
        feature_tokens = feature_tokens + jitter
    return feature_tokens


class RAS(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        initializer,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed = PositionEmbedding(feature_size, hidden_dim // 2)

        input_dim = inplanes[0]
        output_dim = inplanes[0]
        self.transformer = RASformer(
            hidden_dim, output_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        initialize_from_cfg(self, initializer)

    def forward(self, input):
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        if self.training and self.feature_jitter:
            feature_tokens = add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos = self.pos_embed(feature_tokens)  # (H x W) x C
        
        output_decoder, _ = self.transformer(
            feature_tokens, pos
        )  # (H x W) x B x C
        feature_rec = rearrange(
            output_decoder[-1], "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        return {
            "feature_rec":  feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }


class RASformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        feature_size,
        neighbor_mask,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        # att_ras=True,
        # history_ras=True,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        decoder_layer = RASformerDecoderLayer(
            hidden_dim,
            output_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            # att_ras,
            # history_ras,
        )
        self.decoder = RASformerDecoder(
            decoder_layer,
            num_decoder_layers
        )

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nhead = nhead

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, src, pos_embedding):
        _, batch_size, _ = src.shape
        pos_embedding = torch.cat(
            [pos_embedding.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder(
            src, mask=mask_enc, pos=pos_embedding
        )  # (H X W) x B x C
        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embedding,
        )  # [(H X W) x B x C]_{num_decoder_layers}

        return output_decoder, output_encoder


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        # intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            # intermediate.append(output)
        return output
        # return intermediate


class RASformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.forward_layers = _get_clones(decoder_layer, num_layers)
        # self.backward_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

        feature_size = decoder_layer.feature_size
        hidden_dim = decoder_layer.hidden_dim
        num_queries = feature_size[0] * feature_size[1]
        context_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C
        self.embed_layers = _get_clones(context_embed, num_layers)

    def forward(
        self,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        
        hidden = memory
        output = None

        intermediate = []

        for i, layer in enumerate(self.forward_layers):
            context_embed = self.embed_layers[i]
            hidden, output = layer(
                context_embed, 
                hidden,
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            intermediate.append(output)

        return intermediate


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class RASformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        # att_ras=True,
        # history_ras=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dim_feedforward = dim_feedforward
        self.feature_size = feature_size
        self.nhead = nhead
        self.dropout = dropout 
        self.output_transBlock = TransformerBlock(hidden_dim, output_dim, feature_size, nhead, dim_feedforward, dropout=dropout, activation=activation)
        self.adaptive_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.update_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.droupout_proj = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        context_embed,
        latent,
        input,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        c = context_embed.weight
        c = torch.cat([c.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C
        cl = torch.cat([c, latent], dim=2)
        a = torch.sigmoid(self.adaptive_gate(cl))
        l_a = a * latent
        l_T = self.output_transBlock(c, l_a, memory_mask, memory_key_padding_mask, pos)
        l_star = (self.update_proj(latent) + l_T) / 2
        latent = self.output_norm(l_star)
        output = self.output_proj(latent)
        return latent, output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        num_queries = feature_size[0] * feature_size[1]

        self.attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt2 = self.attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbedding(nn.Module):
    def __init__(self, feature_size, num_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        emb = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return emb
