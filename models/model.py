from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoVideoProcessor
from transformers import LongformerTokenizer, LongformerModel
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import VideoMAEImageProcessor, AutoConfig

from utils.vid_extractor import extract_long_video
from models.modules import MultimodalFusionModule, MultiHeadGatedFusion


@dataclass
class MMConfig:
    _claim_pt: str = "roberta-base"
    _long_pt: str = "longformer"
    _vision_pt: str = "clip"
    # _video_pt: str = "videomae"
    _video_pt: str = "vjepa2"

    freeze_text: bool = True
    freeze_long_text: bool = True
    freeze_vision: bool = True
    freeze_video: bool = True
    unfreeze_text_last_n: int = 0
    unfreeze_long_last_n: int = 0

    claim_max_length: int = 128
    evidence_max_length: int = 512

    mfm_d_model: int = 256
    mfm_heads: int = 8
    mfm_out_dim: int = 256
    mfm_dropout: float = 0.1

    fusion_hidden: Tuple[int, ...] = (256,)
    cls_dropout: float = 0.4
    num_classes: int = 3


class MMFactCheckingClassifier(nn.Module):
    def __init__(self, cfg: MMConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self._text_processor, self._text_model = self.text_model(cfg._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(
            cfg._long_pt
        )
        self._image_processor, self._vision_model = self.vision_model(cfg._vision_pt)
        self._video_processor, self._video_model = self.video_model(cfg._video_pt)
        self._apply_freeze_policy()

        text_dim = self._text_model.config.hidden_size
        long_text_dim = self._long_text_model.config.hidden_size
        video_dim = self._video_model.config.hidden_size
        self._video_hidden_dim = video_dim

        self.claim_video_trans = MultimodalFusionModule(
            text_in_dim=text_dim,
            img_in_dim=video_dim,
            d_model=cfg.mfm_d_model,
            n_heads=cfg.mfm_heads,
            out_dim=cfg.mfm_out_dim,
            dropout=cfg.mfm_dropout,
        )

        self.claim_evidence_trans = MultimodalFusionModule(
            text_in_dim=text_dim,
            img_in_dim=long_text_dim,
            d_model=cfg.mfm_d_model,
            n_heads=cfg.mfm_heads,
            out_dim=cfg.mfm_out_dim,
            dropout=cfg.mfm_dropout,
        )

        fusion_in = 2 * cfg.mfm_out_dim
        out_fusion_dim = cfg.fusion_hidden[-1] if len(cfg.fusion_hidden) else fusion_in

        self.fusion_mlp = MultiHeadGatedFusion(
            dim1=cfg.mfm_out_dim,
            dim2=cfg.mfm_out_dim,
            out_dim=out_fusion_dim,
            dropout=cfg.mfm_dropout,
        )

        self.dropout = nn.Dropout(cfg.cls_dropout)
        self.classifier = nn.Linear(2 * cfg.mfm_out_dim, cfg.num_classes)

    def video_model(self, type="videomae"):
        if type == "videomae":
            config = AutoConfig.from_pretrained(
                "MCG-NJU/videomae-base", trust_remote_code=True
            )
            processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            model = AutoModel.from_pretrained(
                "MCG-NJU/videomae-base", config=config, trust_remote_code=True
            )
        elif type == "videomaev2":
            config = AutoConfig.from_pretrained(
                "OpenGVLab/VideoMAEv2-Base", trust_remote_code=True
            )
            processor = VideoMAEImageProcessor.from_pretrained(
                "OpenGVLab/VideoMAEv2-Base"
            )
            model = AutoModel.from_pretrained(
                "OpenGVLab/VideoMAEv2-Base", config=config, trust_remote_code=True
            )
        elif type == "vjepa2":
            model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
            processor = AutoVideoProcessor.from_pretrained(
                "facebook/vjepa2-vitl-fpc64-256"
            )
        else:
            raise ValueError(f"Unknown video model type: {type}")
        return processor, model

    def vision_model(self, type="clip"):
        if type == "clip":
            processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError(f"Unknown vision model type: {type}")
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == "longformer":
            processor = LongformerTokenizer.from_pretrained(
                "allenai/longformer-base-4096"
            )
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        else:
            raise ValueError(f"Unknown long text model: {pt}")
        return processor, model

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad = trainable

    def _unfreeze_last_n_transformer_layers(self, module: nn.Module, n: int) -> None:
        if n <= 0:
            return
        encoder = getattr(module, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def _apply_freeze_policy(self) -> None:
        self._set_module_trainable(self._text_model, not self.cfg.freeze_text)
        self._set_module_trainable(self._long_text_model, not self.cfg.freeze_long_text)
        self._set_module_trainable(self._vision_model, not self.cfg.freeze_vision)
        self._set_module_trainable(self._video_model, not self.cfg.freeze_video)

        # Optional partial unfreezing for text backbones.
        self._unfreeze_last_n_transformer_layers(
            self._text_model, self.cfg.unfreeze_text_last_n
        )
        self._unfreeze_last_n_transformer_layers(
            self._long_text_model, self.cfg.unfreeze_long_last_n
        )

    def _process_videos(self, video_evidence, batch_size, device):
        if not isinstance(video_evidence, list):
            video_evidence = [video_evidence]

        if len(video_evidence) != batch_size:
            print(
                f"video_evidence length {len(video_evidence)} does not match batch_size {batch_size}; padding with zeros."
            )

        video_hidden_dim = self._video_hidden_dim
        all_video_features = []
        all_video_masks = []

        for vid in video_evidence:
            if vid is None or not isinstance(vid, str) or not os.path.exists(vid):
                video_feat = torch.zeros(1, video_hidden_dim)  # [1, D]
                video_mask = torch.zeros(1)  # [1]
            else:
                try:
                    print(f"Extracting features from video: {vid}")
                    video_feat = extract_long_video(
                        self._video_model, self._video_processor, vid
                    )  # Returns [1, T, D]
                    video_feat = video_feat.squeeze(0)
                    video_mask = torch.ones(video_feat.shape[0])  # [T]
                except Exception as e:
                    print(f"Error processing video {vid}: {e}")
                    video_feat = torch.zeros(1, video_hidden_dim)  # [1, D]
                    video_mask = torch.zeros(1)  # [1]

            all_video_features.append(video_feat)  # [T, D]
            all_video_masks.append(video_mask)  # [T]

        while len(all_video_features) < batch_size:
            all_video_features.append(torch.zeros(1, video_hidden_dim))
            all_video_masks.append(torch.zeros(1))

        all_video_features = all_video_features[:batch_size]
        all_video_masks = all_video_masks[:batch_size]

        max_clips = max(feat.shape[0] for feat in all_video_features)

        padded_features = []
        padded_masks = []

        for feat, mask in zip(all_video_features, all_video_masks):
            current_len = feat.shape[0]
            if current_len < max_clips:
                pad_len = max_clips - current_len
                feat = torch.cat([feat, torch.zeros(pad_len, video_hidden_dim)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len)], dim=0)

            padded_features.append(feat)
            padded_masks.append(mask)

        video_tokens = torch.stack(padded_features, dim=0).to(device)  # [B, T, D]
        video_mask = torch.stack(padded_masks, dim=0).to(device)  # [B, T]

        return video_tokens, video_mask

    def forward(self, claim, text_evidence, video_evidence=None, labels=None):
        device = next(self.parameters()).device

        # ---- Process claims ----
        if isinstance(claim, str):
            claims = [claim]
        else:
            claims = [str(c) for c in claim]

        batch_size = len(claims)

        # ---- Process text evidence ----
        evidences = []
        if not isinstance(text_evidence, list):
            text_evidence = [text_evidence]

        sep_token = self._long_text_processor.sep_token or "[SEP]"

        for ev in text_evidence:
            if isinstance(ev, (list, tuple)):
                ev_texts = [
                    str(x).strip()
                    for x in ev
                    if x is not None and str(x).strip() not in ("", "nan", "None")
                ]
                evidences.append(f" {sep_token} ".join(ev_texts) if ev_texts else "")
            else:
                ev_str = str(ev) if ev is not None else ""
                evidences.append(
                    ev_str if ev_str.strip() not in ("nan", "None") else ""
                )

        while len(evidences) < batch_size:
            evidences.append("")

        # ---- Encode claims (short text) ----
        claim_encoded = self._text_processor(
            claims,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.claim_max_length,
        )
        claim_encoded = {k: v.to(device) for k, v in claim_encoded.items()}
        claim_out = self._text_model(**claim_encoded)
        claim_tokens = claim_out.last_hidden_state  # (B, Lc, Dt)
        claim_mask = claim_encoded["attention_mask"]  # (B, Lc)

        # ---- Encode evidence (long text) ----
        text_encoded = self._long_text_processor(
            evidences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.evidence_max_length,
        )
        text_encoded = {k: v.to(device) for k, v in text_encoded.items()}
        long_out = self._long_text_model(**text_encoded)
        evidence_tokens = long_out.last_hidden_state  # (B, Le, Dl)
        evidence_mask = text_encoded["attention_mask"]  # (B, Le)

        # ---- Encode videos ----
        video_tokens, video_mask = self._process_videos(
            video_evidence, batch_size, device
        )  # (B, Tv, Dv), (B, Tv)

        claim_visual_fused = self.claim_video_trans(
            text_tokens=claim_tokens,
            img_tokens=video_tokens,
            text_mask=claim_mask,
            img_mask=video_mask,
        )  # (B, D_mfm)

        claim_evidence_fused = self.claim_evidence_trans(
            text_tokens=claim_tokens,
            img_tokens=evidence_tokens,
            text_mask=claim_mask,
            img_mask=evidence_mask,
        )  # (B, D_mfm)

        h = torch.cat([claim_visual_fused, claim_evidence_fused], dim=-1)
        # h = self.fusion_mlp(claim_visual_fused, claim_evidence_fused)
        h = self.dropout(h)
        logits = self.classifier(h)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        out = {"logits": logits, "probs": probs}
        return out
