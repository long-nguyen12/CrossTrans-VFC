import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import re
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageFilter, ImageOps

DATA_PATH = "data/TRUE_Dataset"


def read_json_dataset(split_dir):
    json_files = list(Path(split_dir).glob("*.json"))

    claims = []
    for json_file in tqdm(json_files, desc=f"Reading {split_dir}"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                claims.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return claims


def get_dataset(path):
    """Load train, val, and test datasets from TRUE_Dataset."""
    train_dir = os.path.join(path, "train_val")
    test_dir = os.path.join(path, "test")
    print(f"Loading dataset from {path}...")

    all_train_data = read_json_dataset(train_dir)
    test_data = read_json_dataset(test_dir)

    np.random.shuffle(all_train_data)

    split_idx = int(len(all_train_data) * 0.8)
    train_data = all_train_data[:split_idx]
    val_data = all_train_data[split_idx:]

    # train_data = all_train_data
    # val_data = all_train_data

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data


def clean_data(text):
    """Clean text data."""
    if text is None or str(text) == "nan":
        return ""
    text = re.sub(r"(<p>|</p>|@)+", "", text)
    return text.strip()


def one_hot(a, num_classes):
    """Convert index to one-hot vector."""
    v = np.zeros(num_classes, dtype=int)
    v[a] = 1
    return v


def extract_evidence_text(evidences_dict):
    """Extract evidence text from the evidences dictionary."""
    evidence_list = []
    if not isinstance(evidences_dict, dict):
        return evidence_list

    # Extract evidence from evidence1, evidence2, ..., evidence9
    for i in range(1, 10):
        key = f"evidence{i}"
        if key in evidences_dict:
            evidence_pair = evidences_dict[key]
            if isinstance(evidence_pair, list) and len(evidence_pair) > 0:
                evidence_text = evidence_pair[0]  # First element is the text
                evidence_list.append(clean_data(evidence_text))

    return evidence_list


def resolve_video_path(claim_id: str, data_path: str = DATA_PATH) -> str:
    if not claim_id:
        return ""

    root = Path(data_path)
    candidates = [
        root / "train_val_video" / f"{claim_id}.mp4",
        root / "test_video" / f"{claim_id}.mp4",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return str(root / "test_video" / f"{claim_id}.mp4")


def resolve_keyframe_path(claim_id: str, data_path: str = DATA_PATH) -> str:
    if not claim_id:
        return ""

    root = Path(data_path)
    candidates = [
        root / "train_val_output" / claim_id,
        root / "test_output" / claim_id,
    ]
    frames = []
    for path in candidates:
        if path.exists():
            frame_path = path / "*.jpeg"
            keyframe_files = glob(str(frame_path))
            if keyframe_files:
                for kf in keyframe_files:
                    frame = Image.open(kf).convert("RGB")
                    frames.append(frame)
                return frames
    return frames


def rating_to_label(rating_str):
    rating_lower = rating_str.lower()

    if rating_lower in ["mostly true", "true", "correct attribution"]:
        return 0
    elif rating_lower in ["false", "mostly false", "mixture", "fake", "miscaptioned"]:
        return 1
    else:
        print(rating_lower)


def encode_one_sample(sample):
    """Encode a single sample from the dataset."""
    # Extract fields from JSON structure
    claim = clean_data(sample.get("claim", ""))
    rating_str = sample.get("rating")
    label_idx = rating_to_label(rating_str)
    content = sample.get("content", "")
    evidences_dict = sample.get("evidences", {})
    text_evidence = extract_evidence_text(evidences_dict)

    video_info = sample.get("video_information", {})

    claim_id = clean_data(video_info.get("video_id", ""))
    video_transcript = clean_data(video_info.get("video_transcript", ""))
    video_url = resolve_video_path(claim_id)
    video_description = clean_data(video_info.get("video_description", ""))
    video_headline = clean_data(video_info.get("video_headline", ""))

    image_evidence = sample.get("image_evidence", [])
    keyframes = resolve_keyframe_path(claim_id)

    encoded_sample = {
        "claim_id": claim_id,
        "claim": claim,
        "content": content,
        "label": torch.tensor(one_hot(label_idx, 2), dtype=torch.float),
        "rating": rating_str,
        "text_evidence": text_evidence,
        "video_transcript": video_transcript,
        "video_url": video_url,
        "video_description": video_description,
        "video_headline": video_headline,
        "image_evidence": image_evidence,
        "url": sample.get("url", ""),
        "content": clean_data(sample.get("content", "")),
        "keyframes": keyframes,
    }

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    """Dataset for claim verification."""

    def __init__(self, claim_data):
        self._data = claim_data
        self._encoded = []

        for d in self._data:
            try:
                self._encoded.append(encode_one_sample(d))
            except Exception as e:
                print(f"Error encoding sample: {e}")

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def collate_claim_verification(batch):
    """Collate function for DataLoader."""
    claim_ids = [item["claim_id"] for item in batch]
    claims = [item["claim"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    contents = [item["content"] for item in batch]
    ratings = [item["rating"] for item in batch]
    text_evidences = [item["text_evidence"] for item in batch]
    video_transcripts = [item["video_transcript"] for item in batch]
    video_urls = [item["video_url"] for item in batch]
    video_descriptions = [item["video_description"] for item in batch]
    video_headlines = [item["video_headline"] for item in batch]
    image_evidences = [item["image_evidence"] for item in batch]
    urls = [item["url"] for item in batch]
    contents = [item["content"] for item in batch]

    return {
        "claim_id": claim_ids,
        "claim": claims,
        "label": labels,
        "content": contents,
        "rating": ratings,
        "text_evidence": text_evidences,
        "video_transcript": video_transcripts,
        "video_url": video_urls,
        "video_description": video_descriptions,
        "video_headline": video_headlines,
        "image_evidence": image_evidences,
        "url": urls,
        "content": contents,
    }


def create_dataloaders(
    path=DATA_PATH,
    batch_size=32,
    shuffle_train=True,
    num_workers=0,
    pin_memory=False,
    limit_samples=None,
):
    train_data, val_data, test_data = get_dataset(path)

    if limit_samples:
        train_data = train_data[:limit_samples]
        val_data = val_data[:limit_samples]
        test_data = test_data[:limit_samples]

    train_dataset = ClaimVerificationDataset(train_data)
    val_dataset = ClaimVerificationDataset(val_data)
    test_dataset = ClaimVerificationDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_claim_verification,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_claim_verification,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_claim_verification,
    )

    return train_loader, val_loader, test_loader
