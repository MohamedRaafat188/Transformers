import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


def create_mask(window_size, displacement, upper_lower, right_left):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if right_left:
        mask = rearrange(mask, "(h w) (h1 w1) -> h w h1 w1", w=window_size, w1=window_size)
        mask[:, :, -displacement:, :-displacement] = float("-inf")
        mask[:, :, :-displacement, -displacement:] = float("-inf")
        mask = rearrange(mask, "h h1 w w1 -> (h w) (h1 w1)")

    return mask


class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Window_Attention(nn.Module):
    def __init__(self, hidden_dim, window_size, num_heads, head_dim, relative_pos_embedding, shifted):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.scale = hidden_dim ** -0.5
        self.inner_dim = num_heads * head_dim
        self.to_qkv = nn.Linear(hidden_dim, self.inner_dim * 3)

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)

            self.upper_lower_mask = nn.Parameter(create_mask(window_size=self.window_size, displacement=displacement, upper_lower=True,
                                                             right_left=False), requires_grad=False)
            self.right_left_mask = nn.Parameter(create_mask(window_size=self.window_size, displacement=displacement, upper_lower=False,
                                                             right_left=True), requires_grad=False)

        if relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.relative_indices = self.relative_indices.long()
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embeddings = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(self.inner_dim, hidden_dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        nw_h = x.shape[1] // self.window_size
        nw_w = x.shape[2] // self.window_size

        q, k, v = map(
            lambda e: rearrange(e, "b (nw_h h1) (nw_w w1) (h d) -> b h (nw_h nw_w) (h1 w1) d",
                                h1=self.window_size, w1=self.window_size, h=self.num_heads, nw_h=nw_h, nw_w=nw_w), qkv)

        att_weights = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        if self.relative_pos_embedding:
            att_weights += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]

        if self.shifted:
            att_weights[:, :, -nw_w:] += self.upper_lower_mask
            att_weights[:, :, nw_w-1::nw_w]
            
        att_weights = att_weights.softmax(dim=-1)
        
        out = einsum("b h w i j, b h w j d -> b h w i d", att_weights, v)
        out = rearrange(out, "b h (nw_h nw_w) (h1 w1) d -> b (nw_h h1) (nw_w w1) (h d)",
                        h1=self.window_size, w1=self.window_size, nw_h=nw_h, nw_w=nw_w)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        return out
    

class Swin_Block(nn.Module):
    def __init__(self, hidden_dim, window_size, num_heads, head_dim, relative_pos_embedding=True, shifted=False):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.window_attention = Window_Attention(
            hidden_dim, window_size, num_heads, head_dim, relative_pos_embedding, shifted)
        self.feedforward = FeedForward(hidden_dim)

    def forward(self, x):
        x += self.window_attention(self.norm(x))
        x += self.feedforward(self.norm(x))
        return x
    

class Patch_Merging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.patch_size = downscaling_factor
        self.to_c = nn.Linear(
            in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        x = rearrange(x, "b c (h h1) (w w1) -> b h w (c h1 w1)",
                      h1=self.patch_size, w1=self.patch_size)
        x = self.to_c(x)
        return x
    

class Stage_Module(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, num_heads, head_dim, downscaling_factor, window_size, relative_pos_embedding):
        super().__init__()
        self.patch_merging = Patch_Merging(
            in_channels, hidden_dim, downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers // 2):
            self.layers.append(nn.ModuleList([
                Swin_Block(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, window_size=window_size,
                           relative_pos_embedding=relative_pos_embedding, shifted=False),
                Swin_Block(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, window_size=window_size,
                           relative_pos_embedding=relative_pos_embedding, shifted=True)
            ]))

    def forward(self, x):
        x = self.patch_merging(x)
        for regular_block, shifted_block in self.layers:  # type: ignore
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class Swin(nn.Module):
    def __init__(self, *, layers=(2, 2, 6, 2), hidden_dim=92, in_channels=3, heads=(3, 6, 12, 24), head_dim=32,
                 relative_pos_embedding=True, downscaling_factors=(4, 2, 2, 2), window_size=7, num_classes=1000) -> None:
        super().__init__()
        self.stage1 = Stage_Module(in_channels, hidden_dim, layers[0], heads[0], head_dim, downscaling_factors[0],
                                   window_size, relative_pos_embedding)
        self.stage2 = Stage_Module(hidden_dim, hidden_dim * 2, layers[1], heads[1], head_dim, downscaling_factors[1],
                                   window_size, relative_pos_embedding)
        self.stage3 = Stage_Module(hidden_dim * 2, hidden_dim * 4, layers[2], heads[2], head_dim, downscaling_factors[2],
                                   window_size, relative_pos_embedding)
        self.stage4 = Stage_Module(hidden_dim * 4, hidden_dim * 8, layers[3], heads[3], head_dim, downscaling_factors[3],
                                   window_size, relative_pos_embedding)
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = x.mean(dim=[2, 3])
        x = self.to_out(x)

        return x
