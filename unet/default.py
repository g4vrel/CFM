from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.modules import (
    get_timestep_embedding,
    Downsample,
    Upsample,
    ResBlock,
)


class Unet(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mul: List[int] = [1, 2, 2, 2],
        att_channels: List[int] = [0, 1, 1, 0],
        groups: int = 32,
        dropout: float = 0.1,
        scale_shift: bool = True,
    ):
        super().__init__()
        assert len(att_channels) == len(ch_mul), (
            "Attention bool must be defined for each channel"
        )

        self.ch = ch
        self.ch_mul = ch_mul
        self.att_channels = att_channels
        self.dropout = dropout
        self.scale_shift = scale_shift
        self.groups = groups

        self.temb_dim = self.ch * 4

        self.input_proj = nn.Conv2d(3, self.ch, 3, 1, 1)

        self.time_proj = nn.Sequential(
            nn.Linear(self.ch, self.temb_dim),
            nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
        )

        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])

        self.make_paths()

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=2 * self.ch),
            nn.SiLU(),
            nn.Conv2d(2 * self.ch, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels=None):
        assert t.shape == (x.shape[0],), "t should be a (batch_size,)-shaped array"

        temb = get_timestep_embedding(t, self.ch)
        emb = self.time_proj(temb)

        x = self.input_proj(x)
        h = x.clone()

        down_path = []
        for i in range(len(self.down)):
            h = self.down[i][0](h, emb)
            down_path.append(h)

            h = self.down[i][1](h, emb)
            down_path.append(h)

            if i < (len(self.down) - 1):  # downsample
                h = self.down[i][2](h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h, emb)

        for i in range(len(self.up)):
            h = self.up[i][0](torch.cat((h, down_path.pop()), dim=1), emb)
            h = self.up[i][1](torch.cat((h, down_path.pop()), dim=1), emb)

            if i < (len(self.down) - 1):  # upsample
                h = self.up[i][2](h)

        x = torch.cat((h, x), dim=1)
        return self.final(x)

    def make_transition(self, res, down):
        dim = self.ch * self.ch_mul[res]

        if down:
            is_last_res = res == (len(self.ch_mul) - 1)
            if is_last_res:
                return Downsample(dim, dim)

            dim_out = self.ch * self.ch_mul[res + 1]
            return Downsample(dim, dim_out)

        is_first_res = res == 0
        if is_first_res:
            return Upsample(dim, dim)

        dim_out = self.ch * self.ch_mul[res - 1]
        return Upsample(dim, dim_out)

    def make_res(self, res, down):
        attn = self.att_channels[res] == 1
        dim = self.ch * self.ch_mul[res]
        transition = self.make_transition(res, down)

        if down:
            dim_in = dim
        else:
            dim_in = 2 * dim

        return nn.ModuleList(
            [
                ResBlock(
                    dim_in,
                    dim,
                    self.temb_dim,
                    self.scale_shift,
                    self.groups,
                    self.dropout,
                    attn=attn,
                ),
                ResBlock(
                    dim_in,
                    dim,
                    self.temb_dim,
                    self.scale_shift,
                    self.groups,
                    self.dropout,
                    attn=attn,
                ),
                transition,
            ]
        )

    def make_paths(self):
        num_res = len(self.ch_mul)

        for res in range(num_res):
            is_last_res = res == (num_res - 1)

            down_blocks = self.make_res(res, down=True)
            up_blocks = self.make_res(res, down=False)

            if is_last_res:
                down_blocks = down_blocks[:-1]
            if res == 0:
                up_blocks = up_blocks[:-1]

            self.down.append(down_blocks)
            self.up.insert(0, up_blocks)

        nch = self.ch * self.ch_mul[-1]
        self.mid = nn.ModuleList(
            [
                ResBlock(
                    nch,
                    nch,
                    self.temb_dim,
                    self.scale_shift,
                    self.groups,
                    self.dropout,
                    attn=True,
                ),
                ResBlock(
                    nch,
                    nch,
                    self.temb_dim,
                    self.scale_shift,
                    self.groups,
                    self.dropout,
                    attn=False,
                ),
            ]
        )
