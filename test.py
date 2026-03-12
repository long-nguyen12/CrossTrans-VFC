import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from models.model import MMConfig, MMFactCheckingClassifier
from utils.true_dataset import create_dataloaders


def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.argmax(dim=-1)


def load_checkpoint_compat(path: Path, device: torch.device):
    """Compatibility loader for PyTorch>=2.6 default weights_only=True."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_cfg_from_checkpoint(checkpoint: Dict) -> MMConfig:
    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise KeyError("Checkpoint does not contain 'cfg'.")

    valid_fields = {f.name for f in fields(MMConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return MMConfig(**filtered)


@torch.no_grad()
def evaluate(model, loader, device: torch.device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)

    y_true, y_pred = [], []
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="Testing"):
        labels = normalize_labels(batch["label"]).to(device)
        out = model(
            claim=batch["claim"],
            text_evidence=batch["content"],
            video_evidence=batch["video_url"],
            labels=labels,
        )
        logits = out["logits"]
        loss = loss_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    loss = total_loss / max(n, 1)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return loss, acc, precision, recall, f1, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints\\best\\best.pt",
        help="Path to .pt file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../TRUE-3MFact/data/TRUE_Dataset",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output", type=str, default="", help="Optional JSON output path"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint_compat(checkpoint_path, device)

    cfg = build_cfg_from_checkpoint(checkpoint)
    model = MMFactCheckingClassifier(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    label2id = checkpoint.get("label2id", {"TRUE": 0, "FALSE": 1})
    id2label = {v: k for k, v in label2id.items()}

    train_loader, val_loader, test_loader = create_dataloaders(
        path=args.data_root,
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluating split: {args.split}")
    print(f"Samples: {len(loader.dataset)}")

    loss, acc, precision, recall, f1, y_true, y_pred = evaluate(model, loader, device)

    print("\nResults:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy (avg): {acc:.4f}")
    print(f"  Precision (macro avg): {precision:.4f}")
    print(f"  Recall (macro avg): {recall:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(label2id))],
            digits=4,
            zero_division=0,
        )
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "loss": float(loss),
            "accuracy": float(acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "n_samples": len(loader.dataset),
            "predictions": y_pred,
            "true_labels": y_true,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved output to: {out_path}")


if __name__ == "__main__":
    main()
