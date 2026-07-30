import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

# --- Interpolation --------------------------------------------------------
# Position / relative-position embeddings are resized to the current grid with
# the same algorithm as the reference torch F.interpolate (bicubic+antialias /
# linear). PIL 'F'-mode resize is bit-for-bit equivalent, so we build the
# separable [out, in] resize operators by probing PIL with unit impulses (cached
# per size), then apply them to every channel with a single matmul in float32.


@lru_cache(maxsize=64)
def _resize_matrix(in_size, out_size, mode):
    """[out_size, in_size] resize operator (align_corners=False).

    ``linear`` uses the exact analytic weights (used for rel_pos upsampling);
    ``bicubic`` probes PIL, which is bit-identical to torch's antialiased
    bicubic (used for positional-embedding resizing at non-native grids).
    """
    R = np.zeros((out_size, in_size), dtype=np.float32)
    if mode == "linear":
        pos = (np.arange(out_size, dtype=np.float32) + 0.5) * (in_size / out_size) - 0.5
        lo = np.clip(np.floor(pos).astype(np.int32), 0, in_size - 1)
        hi = np.clip(lo + 1, 0, in_size - 1)
        frac = pos - np.floor(pos)
        for i in range(out_size):
            R[i, lo[i]] += 1.0 - frac[i]
            R[i, hi[i]] += frac[i]
    else:  # bicubic
        col = np.zeros((in_size, 1), dtype=np.float32)
        for j in range(in_size):
            col[:] = 0.0
            col[j, 0] = 1.0
            R[:, j] = np.asarray(
                Image.fromarray(col, mode="F").resize((1, out_size), Image.BICUBIC)
            )[:, 0]
    return mx.array(R)


def _resize_nchw(x, out_h, out_w, mode):
    """Resize [B, C, H, W] -> [B, C, out_h, out_w] channel-wise (float32)."""
    dtype = x.dtype
    x = x.astype(mx.float32)
    Rh = _resize_matrix(x.shape[2], out_h, mode)
    Rw = _resize_matrix(x.shape[3], out_w, mode)
    x = mx.einsum("oh,bchw->bcow", Rh, x)
    x = mx.einsum("vw,bcow->bcov", Rw, x)
    return x.astype(dtype)


def _resize_axis0(x, out_size, mode):
    """Resize [L, C] -> [out_size, C] along axis 0 (float32)."""
    R = _resize_matrix(x.shape[0], out_size, mode)
    return (R @ x.astype(mx.float32)).astype(x.dtype)


def quick_gelu(x):
    return x * mx.sigmoid(1.702 * x)


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.shape[-1]
    abs_pos = abs_pos.reshape(-1, dim)
    cls_token = abs_pos[:1]
    old_pos_embed = abs_pos[1:]

    src_size = int(math.sqrt(old_pos_embed.shape[0]))
    tgt_size_int = int(math.sqrt(tgt_size))

    if src_size != tgt_size_int:
        old_pos_embed = old_pos_embed.reshape(1, src_size, src_size, dim).transpose(
            0, 3, 1, 2
        )
        new_pos_embed = _resize_nchw(
            old_pos_embed, tgt_size_int, tgt_size_int, "bicubic"
        )
        new_pos_embed = new_pos_embed.transpose(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.reshape(tgt_size_int * tgt_size_int, dim)
        vision_pos_embed = mx.concatenate([cls_token, new_pos_embed], axis=0)
        return vision_pos_embed.reshape(1, tgt_size_int * tgt_size_int + 1, dim)
    else:
        return abs_pos.reshape(1, -1, dim)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = mx.zeros((self.embed_dim,))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, pixel_values, patch_embeds=None):
        batch_size = pixel_values.shape[0]

        if patch_embeds is not None:
            patch_embeds_out = patch_embeds
        else:
            pixel_values_nhwc = pixel_values.transpose(0, 2, 3, 1)
            patch_embeds_out = self.patch_embedding(pixel_values_nhwc)

        patch_embeds_out = patch_embeds_out.reshape(
            batch_size, self.embed_dim, -1
        ).transpose(0, 2, 1)

        class_embeds = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, self.embed_dim)
        )
        embeddings = mx.concatenate([class_embeds, patch_embeds_out], axis=1)

        pos_ids = mx.arange(self.num_positions).reshape(1, -1)
        pos_embeds = self.position_embedding(pos_ids)
        pos_embeds = get_abs_pos(pos_embeds, embeddings.shape[1])
        embeddings = embeddings + pos_embeds
        return embeddings


class NoTPFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class NoTPAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x):
        bsz, seqlen, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        output = self.out_proj(output)
        return output


class NoTPTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden_size, layernorm_epsilon=1e-5):
        super().__init__()
        self.self_attn = NoTPAttention(hidden_size, num_heads)
        self.mlp = NoTPFeedForward(hidden_size, ffn_hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

    def __call__(self, x):
        h = x + self.self_attn(self.layer_norm1(x))
        out = h + self.mlp(self.layer_norm2(h))
        return out


class NoTPTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_heads,
        ffn_hidden_size,
        layernorm_epsilon=1e-5,
    ):
        super().__init__()
        self.layers = [
            NoTPTransformerBlock(
                hidden_size, num_heads, ffn_hidden_size, layernorm_epsilon
            )
            for _ in range(num_layers)
        ]

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class VitModel(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        image_size=224,
        patch_size=14,
        num_layers=24,
        num_heads=16,
        ffn_hidden_size=4096,
        layernorm_epsilon=1e-5,
        pre_layernorm_epsilon=1e-5,
    ):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=hidden_size, image_size=image_size, patch_size=patch_size
        )
        self.transformer = NoTPTransformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_hidden_size=ffn_hidden_size,
            layernorm_epsilon=layernorm_epsilon,
        )
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=pre_layernorm_epsilon)

    def __call__(self, x, patch_embeds=None):
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)
        output = self.transformer(hidden_states)
        return output


def build_clip_l():
    return VitModel(
        hidden_size=1024,
        image_size=224,
        patch_size=14,
        num_layers=24,
        num_heads=16,
        ffn_hidden_size=4096,
        layernorm_epsilon=1e-5,
        pre_layernorm_epsilon=1e-5,
    )


def get_abs_pos_sam(abs_pos, tgt_size):
    src_size = abs_pos.shape[1]
    if src_size != tgt_size:
        old_pos_embed = abs_pos.transpose(0, 3, 1, 2)
        new_pos_embed = _resize_nchw(old_pos_embed, tgt_size, tgt_size, "bicubic")
        new_pos_embed = new_pos_embed.transpose(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x):
        u = x.mean(axis=1, keepdims=True)
        s = ((x - u) ** 2).mean(axis=1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)
        x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(1, -1, 1, 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def __call__(self, x):
        return self.lin2(nn.gelu(self.lin1(x)))


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = _resize_axis0(rel_pos, max_rel_dist, "linear")
    else:
        rel_pos_resized = rel_pos

    q_coords = mx.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = mx.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.astype(mx.int32)]


def add_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = mx.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = mx.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def __call__(self, x):
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q.reshape(B * self.num_heads, H * W, -1),
                self.rel_pos_h,
                self.rel_pos_w,
                (H, W),
                (H, W),
            )
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)
            v = v.reshape(B, self.num_heads, H * W, -1)
            rel_h = rel_h.reshape(
                B, self.num_heads, rel_h.shape[1], rel_h.shape[2], rel_h.shape[3]
            )
            rel_w = rel_w.reshape(
                B, self.num_heads, rel_w.shape[1], rel_w.shape[2], rel_w.shape[3]
            )
            attn_bias = (rel_h + rel_w).reshape(
                B, self.num_heads, rel_h.shape[2], rel_h.shape[3] * rel_w.shape[4]
            )
            x = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=attn_bias
            )
        else:
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)
            v = v.reshape(B, self.num_heads, H * W, -1)
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = (
            x.reshape(B, self.num_heads, H, W, -1)
            .transpose(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def __call__(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size=(16, 16),
        stride=(16, 16),
        in_chans=3,
        embed_dim=768,
        bias=False,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),
            bias=bias,
        )

    def __call__(self, x):
        x = self.proj(x)
        return x


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        global_attn_indexes=(),
        patch_embed_bias=False,
    ):
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=patch_embed_bias,
        )

        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = mx.zeros(
                (1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = []
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = [
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        ]

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos_sam(self.pos_embed, x.shape[1])

        for blk in self.blocks:
            x = blk(x)

        for layer in self.neck:
            if isinstance(layer, LayerNorm2d):
                x = x.transpose(0, 3, 1, 2)
                x = layer(x)
                x = x.transpose(0, 2, 3, 1)
            else:
                x = layer(x)
        x2 = self.net_2(x)
        x3 = self.net_3(x2)

        return x3.transpose(0, 3, 1, 2)


def build_sam_vit_b():
    return ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=(2, 5, 8, 11),
        window_size=14,
        out_chans=256,
        patch_embed_bias=True,
    )
