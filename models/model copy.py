from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTImageProcessor, ViTModel, BigBirdModel, BigBirdTokenizer
from transformers import BeitImageProcessor, BeitModel, DeiTModel, DeiTImageProcessor
from transformers import AutoTokenizer, AutoModel
from transformers import LongformerTokenizer, LongformerModel
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPTokenizer,
    CLIPTextModel,
)
from transformers import VideoMAEForVideoClassification

from PIL import Image


def masked_mean(
    x: torch.Tensor, mask: Optional[torch.Tensor], dim: int
) -> torch.Tensor:
    """
    x: (B, L, D)
    mask: (B, L) with 1 for valid positions, 0 for pad
    """
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(x.dtype)

    if mask.shape[1] != x.shape[1]:
        raise ValueError(
            f"Mask sequence length {mask.shape[1]} doesn't match x sequence length {x.shape[1]}. "
            f"x shape: {x.shape}, mask shape: {mask.shape}"
        )

    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)  # (B, 1)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        dims = (in_dim,) + hidden_dims + (out_dim,)
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), act, nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Multimodal Fusion Module
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=k_padding_mask,
            need_weights=False,
        )
        x = self.norm1(q + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class MultimodalFusionModule(nn.Module):
    """Enhanced multimodal fusion with bi-directional cross-attention."""

    def __init__(
        self,
        text_in_dim: int,
        img_in_dim: int,
        d_model: int,
        n_heads: int,
        out_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.text_to_model = nn.Linear(text_in_dim, d_model)
        self.img_to_model = nn.Linear(img_in_dim, d_model)

        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "t_from_i": CrossAttention(d_model, n_heads, dropout),
                        "i_from_t": CrossAttention(d_model, n_heads, dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

        self.fuse_mlp = MLP(
            in_dim=2 * d_model,
            hidden_dims=(d_model,),
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = self.text_to_model(text_tokens)  # (B, Lt, D)
        i = self.img_to_model(img_tokens)  # (B, Li, D)

        img_kpm = None if img_mask is None else (img_mask == 0)
        txt_kpm = None if text_mask is None else (text_mask == 0)

        for layer in self.cross_attn_layers:
            t = layer["t_from_i"](q=t, k=i, v=i, k_padding_mask=img_kpm)
            i = layer["i_from_t"](q=i, k=t, v=t, k_padding_mask=txt_kpm)

        t_vec = masked_mean(t, text_mask, dim=1)
        i_vec = masked_mean(i, img_mask, dim=1)
        fused = torch.cat([t_vec, i_vec], dim=-1)
        return self.fuse_mlp(fused)


class MultiHeadGatedFusion(nn.Module):
    def __init__(
        self,
        dim1: int,
        dim2: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.proj1 = nn.ModuleList(
            [nn.Linear(dim1, self.head_dim) for _ in range(num_heads)]
        )
        self.proj2 = nn.ModuleList(
            [nn.Linear(dim2, self.head_dim) for _ in range(num_heads)]
        )

        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim1 + dim2, self.head_dim), nn.Sigmoid())
                for _ in range(num_heads)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1: (B, dim1), x2: (B, dim2)"""
        combined_input = torch.cat([x1, x2], dim=-1)

        head_outputs = []
        for i in range(self.num_heads):
            h1 = self.proj1[i](x1)
            h2 = self.proj2[i](x2)
            gate = self.gates[i](combined_input)
            head_out = gate * h1 + (1 - gate) * h2
            head_outputs.append(head_out)

        fused = torch.cat(head_outputs, dim=-1)  # (B, out_dim)
        return self.norm(self.dropout(fused))


@dataclass
class MMConfig:
    _claim_pt: str = "roberta-base"
    _long_pt: str = "longformer"
    _vision_pt: str = "clip"
    _video_pt: str = "videomae"

    freeze_bert: bool = True
    freeze_clip: bool = True

    claim_max_length: int = 128
    evidence_max_length: int = 512

    mfm_d_model: int = 256
    mfm_heads: int = 8
    mfm_out_dim: int = 256
    mfm_dropout: float = 0.1

    # Final
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

        self._text_clip = None

        text_dim = self._text_model.config.hidden_size
        long_text_dim = self._long_text_model.config.hidden_size
        img_dim = self._vision_model.config.hidden_size

        # Modules
        self.claim_image_trans = MultimodalFusionModule(
            text_in_dim=text_dim,
            img_in_dim=img_dim,
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
        self.classifier = nn.Linear(out_fusion_dim, cfg.num_classes)

        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if n.startswith("head"):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def video_model(self, type="videomae"):
        if type == "videomae":
            model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics",
                attn_implementation="sdpa",
                dtype=torch.float16,
            )
        else:
            raise ValueError(f"Unknown video model type: {type}")

        model.requires_grad_(False)
        return model

    def vision_model(self, type="vit"):
        if type == "vit":
            processor = ViTImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k"
            )
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif type == "beit":
            processor = BeitImageProcessor.from_pretrained(
                "microsoft/beit-base-patch16-224-pt22k"
            )
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        elif type == "deit":
            processor = DeiTImageProcessor.from_pretrained(
                "facebook/deit-base-distilled-patch16-224"
            )
            model = DeiTModel.from_pretrained(
                "facebook/deit-base-distilled-patch16-224"
            )
        elif type == "clip":
            processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError(f"Unknown vision model type: {type}")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        if pt == "roberta-base":
            processor = AutoTokenizer.from_pretrained(pt)
            model = AutoModel.from_pretrained(pt)
        elif pt == "clip":
            processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        model.requires_grad_(False)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == "longformer":
            processor = LongformerTokenizer.from_pretrained(
                "allenai/longformer-base-4096"
            )
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        elif pt == "bigbird":
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        else:
            raise ValueError(f"Unknown long text model type: {pt}")

        model.requires_grad_(False)
        return processor, model

    def _process_images(self, image_evidence, batch_size):
        images = []

        if not isinstance(image_evidence, list):
            image_evidence = [image_evidence]

        for img in image_evidence:
            if img is None:
                images.append(Image.new("RGB", (224, 224), color="white"))
            elif isinstance(img, str):
                try:
                    images.append(Image.open(img).convert("RGB"))
                except:
                    images.append(Image.new("RGB", (224, 224), color="white"))
            elif isinstance(img, Image.Image):
                images.append(img.convert("RGB"))
            else:
                images.append(Image.new("RGB", (224, 224), color="white"))

        while len(images) < batch_size:
            images.append(Image.new("RGB", (224, 224), color="white"))

        return images[:batch_size]

    def forward(
        self,
        claim: torch.Tensor,
        text_evidence: torch.Tensor,
        image_evidence: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # (B,)
    ) -> Dict[str, torch.Tensor]:
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

        for ev in text_evidence:
            if isinstance(ev, (list, tuple)):
                ev_texts = [
                    str(x).strip()
                    for x in ev
                    if x is not None and str(x).strip() not in ("", "nan", "None")
                ]
                evidences.append(" [SEP] ".join(ev_texts) if ev_texts else "")
            else:
                ev_str = str(ev) if ev is not None else ""
                evidences.append(
                    ev_str if ev_str.strip() not in ("nan", "None") else ""
                )

        while len(evidences) < batch_size:
            evidences.append("")

        # ---- Process image evidence ----
        images = self._process_images(image_evidence, batch_size)

        # ---- Encode claims (short text) ----
        claim_encoded = self._text_processor(
            claims,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=(
                self.cfg.claim_max_length
                if hasattr(self.cfg, "claim_max_length")
                else 100
            ),
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
            max_length=(
                self.cfg.evidence_max_length
                if hasattr(self.cfg, "evidence_max_length")
                else 512
            ),
        )
        text_encoded = {k: v.to(device) for k, v in text_encoded.items()}
        long_out = self._long_text_model(**text_encoded)
        evidence_tokens = long_out.last_hidden_state  # (B, Le, Dl)
        evidence_mask = text_encoded["attention_mask"]  # (B, Le)

        # ---- Encode images ----
        image_encoded = self._image_processor(
            images,
            return_tensors="pt",
        )
        image_encoded = {k: v.to(device) for k, v in image_encoded.items()}
        vision_out = self._vision_model(**image_encoded)
        img_tokens = vision_out.last_hidden_state  # (B, Li, Di)
        img_mask = None

        # ---- Multimodal fusion ----
        claim_visual_fused = self.claim_image_trans(
            text_tokens=claim_tokens,
            img_tokens=img_tokens,
            text_mask=claim_mask,
            img_mask=img_mask,
        )  # (B, D_mfm)

        evidence_visual_fused = self.claim_evidence_trans(
            text_tokens=claim_tokens,
            img_tokens=evidence_tokens,
            text_mask=claim_mask,
            img_mask=evidence_mask,
        )  # (B, D_mfm)

        # fused = torch.cat([claim_visual_fused, evidence_visual_fused], dim=-1)
        h = self.fusion_mlp(claim_visual_fused, evidence_visual_fused)
        h = self.dropout(h)
        logits = self.classifier(h)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        out = {
            "logits": logits,
            "probs": probs,
            "claim_visual_fused": claim_visual_fused,
            "evidence_visual_fused": evidence_visual_fused,
        }

        return out
