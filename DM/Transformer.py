import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch
import torch.nn as nn
import math


@dataclass
class Config:
    d_residual: int = 64
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 100
    init_range: float = None
    max_length: int = 256
    d_head: int = 32
    d_mlp: int = 4 * 64
    n_heads: int = 2
    n_layers: int = 2


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_residual))
        self.b = nn.Parameter(torch.zeros(cfg.d_residual))
    
    def forward(self, residual):
        # residual: [batch, position, d_residual]
        if self.cfg.debug: print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual, "batch position d_residual -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_residual -> batch position 1", "mean") + self.cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_residual)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_E, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_E)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :] # [batch, position, d_residual]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed


class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.max_length, cfg.d_residual)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_pos, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_pos)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_residual]
        pos_embed = einops.repeat(pos_embed, "position d_residual -> batch position d_residual", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_residual, cfg.d_head)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_Q)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_residual, cfg.d_head)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_K, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_K)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_residual, cfg.d_head)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_V, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_V)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_residual)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_O, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_O)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_residual)))
        
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"))
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_residual]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)
        
        q = einsum("batch query_pos d_residual, n_heads d_residual d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_residual, n_heads d_residual d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        self.attention = pattern

        v = einsum("batch key_pos d_residual, n_heads d_residual d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_residual -> batch query_pos d_residual", z, self.W_O) + self.b_O
        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_residual, cfg.d_mlp)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_in, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_in)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_residual)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_out, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_out)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_residual)))
    
    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = torch.nn.GELU()(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attn_out
        
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_residual, cfg.d_vocab)))
        if self.cfg.init_range is not None:
            nn.init.normal_(self.W_U, std=self.cfg.init_range)
        else:
            nn.init.kaiming_normal_(self.W_U)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))
    
    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_residual]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_residual, d_residual d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits