from einops import rearrange, repeat
import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, do_p: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=do_p),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, num_heads: int, dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.keys = nn.Linear(dim, dim, bias=False)
        self.queries = nn.Linear(dim, dim, bias=False)
        self.values = nn.Linear(dim, dim, bias=False)

        self.sm = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)

        self.to_out = nn.Linear(dim, dim)


    def forward(self, x):
        x = self.norm(x)

        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h = self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        att_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

        att_weights = self.sm(att_weights)

        out = torch.matmul(att_weights, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
    

class Transformer(nn.Module):
    def __init__(self, depth: int, heads: int, dim: int, hidden_dim: int, do_p: float) -> None:
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(heads, dim),
                FeedForward(dim, hidden_dim, do_p),
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for att, ff in self.layers:  # type: ignore
            x += att(x)
            x += ff(x)

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, channels: int, dim: int, emb_dropout: float, depth: int, num_heads: int, hidden_dim: int, trans_p, num_classes):
        super().__init__()
        img_height, img_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert img_height % patch_height == 0 and img_width % patch_width == 0, "Img dimensions must be divisible by patch dimensions"

        num_patches = (img_height // patch_height) * (img_width // patch_width)
        patch_dim = channels * patch_width * patch_height
        self.to_patch_embeddings = nn.Sequential(
            Rearrange("b c (h h1) (w w1) -> b (h w) (h1 w1 c)",
                      h1=patch_height, w1=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(p=emb_dropout)

        self.transformer = Transformer(
            depth, num_heads, dim, hidden_dim, trans_p)

        self.to_out = nn.Linear(dim, num_classes)

        self.sm = nn.Softmax(dim=1)

        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.to_patch_embeddings(x)
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embeddings

        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0]

        out = self.to_latent(x)
        x = self.to_out(out)

        return self.sm(x)
