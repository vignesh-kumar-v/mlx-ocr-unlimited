"""DeepSeek-OCR: SAM ViT-B + CLIP ViT-L vision encoders, a linear projector, and
a DeepSeek-V2 MoE language model."""

import mlx.core as mx
import mlx.nn as nn

from .cache import PrefillWindowCache
from .config import UnlimitedOCRConfig
from .deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2RMSNorm
from .projector import MlpProjector
from .vision_encoder import build_clip_l, build_sam_vit_b


def _combine_vision_features(sam_model, vision_model, projector, pixel_values):
    """SAM + CLIP features for one batch of images -> projected [N, hw, n_embed]."""
    feat_sam = sam_model(pixel_values)
    feat_clip = vision_model(pixel_values, feat_sam)
    feat = mx.concatenate(
        [
            feat_clip[:, 1:],
            feat_sam.reshape(feat_sam.shape[0], feat_sam.shape[1], -1).transpose(
                0, 2, 1
            ),
        ],
        axis=-1,
    )
    return projector(feat)


class UnlimitedOCRModel(nn.Module):
    def __init__(self, config: UnlimitedOCRConfig):
        super().__init__()
        self.config = config
        lang_config = config.language_config

        self.embed_tokens = nn.Embedding(
            lang_config.vocab_size, lang_config.hidden_size
        )
        self.layers = [
            DeepseekV2DecoderLayer(lang_config, i)
            for i in range(lang_config.num_hidden_layers)
        ]
        self.norm = DeepseekV2RMSNorm(
            lang_config.hidden_size, eps=lang_config.rms_norm_eps
        )

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        n_embed = config.projector_config.n_embed
        self.projector = MlpProjector(
            projector_type=config.projector_config.projector_type,
            input_dim=config.projector_config.input_dim,
            n_embed=n_embed,
        )
        self.image_newline = mx.zeros((n_embed,))
        self.view_seperator = mx.zeros((n_embed,))

    def _embed_images(
        self, inputs_embeds, images, images_seq_mask, images_spatial_crop
    ):
        newline = self.image_newline
        sep = self.view_seperator
        for batch_idx in range(inputs_embeds.shape[0]):
            patches, image_ori = images[batch_idx]
            seq_mask = images_seq_mask[batch_idx]

            if mx.abs(patches).sum() > 0:
                # Crop ("Gundam") mode: global view + local tiles.
                crop_shape = images_spatial_crop[batch_idx]
                w_crop, h_crop = int(crop_shape[0].item()), int(crop_shape[1].item())

                local = _combine_vision_features(
                    self.sam_model, self.vision_model, self.projector, patches
                )
                glob = _combine_vision_features(
                    self.sam_model, self.vision_model, self.projector, image_ori
                )
                n_dim = glob.shape[-1]

                h = w = int(glob.shape[1] ** 0.5)
                glob = glob.reshape(h, w, n_dim)
                glob = mx.concatenate(
                    [
                        glob,
                        mx.broadcast_to(newline.reshape(1, 1, n_dim), (h, 1, n_dim)),
                    ],
                    axis=1,
                )
                glob = glob.reshape(-1, n_dim)

                h2 = w2 = int(local.shape[1] ** 0.5)
                local = local.reshape(h_crop, w_crop, h2, w2, n_dim).transpose(
                    0, 2, 1, 3, 4
                )
                local = local.reshape(h_crop * h2, w_crop * w2, n_dim)
                local = mx.concatenate(
                    [
                        local,
                        mx.broadcast_to(
                            newline.reshape(1, 1, n_dim), (h_crop * h2, 1, n_dim)
                        ),
                    ],
                    axis=1,
                )
                local = local.reshape(-1, n_dim)

                features = mx.concatenate([local, glob, sep.reshape(1, n_dim)], axis=0)
            else:
                # No-crop mode: each image is a global view separated by view_seperator.
                parts = []
                for img_idx in range(image_ori.shape[0]):
                    glob = _combine_vision_features(
                        self.sam_model,
                        self.vision_model,
                        self.projector,
                        image_ori[img_idx : img_idx + 1],
                    )
                    n_dim = glob.shape[-1]
                    h = w = int(glob.shape[1] ** 0.5)
                    glob = glob.reshape(h, w, n_dim)
                    glob = mx.concatenate(
                        [
                            glob,
                            mx.broadcast_to(
                                newline.reshape(1, 1, n_dim), (h, 1, n_dim)
                            ),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim)
                    parts.append(glob)
                    parts.append(sep.reshape(1, n_dim))
                features = mx.concatenate(parts, axis=0)

            valid = [i for i, m in enumerate(seq_mask.tolist()) if m]
            n = min(len(valid), features.shape[0])
            if n > 0:
                inputs_embeds[batch_idx, valid[:n]] = features[:n].astype(
                    inputs_embeds.dtype
                )
        return inputs_embeds

    def __call__(
        self,
        input_ids=None,
        inputs_embeds=None,
        cache=None,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if images is not None and images_seq_mask is not None:
            inputs_embeds = self._embed_images(
                inputs_embeds, images, images_seq_mask, images_spatial_crop
            )

        mask = "causal" if inputs_embeds.shape[1] > 1 else None
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i] if cache is not None else None)
        return self.norm(h)


class UnlimitedOCRForCausalLM(nn.Module):
    def __init__(self, config: UnlimitedOCRConfig):
        super().__init__()
        self.config = config
        self.model = UnlimitedOCRModel(config)
        self.lm_head = nn.Linear(
            config.language_config.hidden_size,
            config.language_config.vocab_size,
            bias=False,
        )

    def __call__(
        self,
        input_ids=None,
        inputs_embeds=None,
        cache=None,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
    ):
        h = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )
        return self.lm_head(h)

    def make_cache(self):
        window = self.config.language_config.sliding_window_size
        return [PrefillWindowCache(window) for _ in self.model.layers]
