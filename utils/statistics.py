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

from true_dataset import create_dataloaders, get_dataset

DATA_PATH = "data/TRUE_Dataset"


def compute_dataset_statistics(data, split_name="Dataset"):
    """Compute comprehensive statistics for a dataset split."""
    stats = {
        "split": split_name,
        "total_samples": len(data),
        "label_distribution": Counter(),
        "rating_distribution": Counter(),
        "claim_lengths": [],
        "evidence_counts": [],
        "content_lengths": [],
        "video_transcript_lengths": [],
        "has_video": 0,
        "has_image": 0,
        "has_evidence": 0,
    }

    for sample in data:
        rating_str = sample.get("rating")
        stats["rating_distribution"][rating_str] += 1

        label_idx = rating_to_label(rating_str)
        label_name = ["TRUE", "FALSE"][label_idx]
        stats["label_distribution"][label_name] += 1

        # Text lengths
        claim = sample.get("claim", "")
        stats["claim_lengths"].append(len(claim.split()))

        content = sample.get("content", "")
        stats["content_lengths"].append(len(content.split()))

        # Evidence count
        evidences_dict = sample.get("evidences", {})
        evidence_list = extract_evidence_text(evidences_dict)
        stats["evidence_counts"].append(len(evidence_list))
        if len(evidence_list) > 0:
            stats["has_evidence"] += 1

        # Video information
        video_info = sample.get("video_information", {})
        if video_info:
            stats["has_video"] += 1
            video_transcript = video_info.get("video_transcript", "")
            stats["video_transcript_lengths"].append(len(video_transcript.split()))
        else:
            print(video_info)

        # Image information
        image_evidence = sample.get("evidences", [])
        if image_evidence and image_evidence.get("num_of_evidence", 0) > 0:
            stats["has_image"] += 1

    return stats


def print_statistics(stats):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"Statistics for {stats['split']}")
    print(f"{'='*60}")

    print(f"\nTotal Samples: {stats['total_samples']}")

    print(f"\nLabel Distribution:")
    for label, count in stats["label_distribution"].most_common():
        percentage = (count / stats["total_samples"]) * 100
        print(f"  {label:12s}: {count:6d} ({percentage:5.2f}%)")

    print(f"\nRating Distribution (Top 10):")
    for rating, count in stats["rating_distribution"].most_common(10):
        percentage = (count / stats["total_samples"]) * 100
        print(f"  {rating:25s}: {count:6d} ({percentage:5.2f}%)")

    print(f"\nText Length Statistics (words):")
    print(f"  Claim length:")
    print(
        f"    Mean: {np.mean(stats['claim_lengths']):.2f}, Median: {np.median(stats['claim_lengths']):.2f}"
    )
    print(
        f"    Min: {np.min(stats['claim_lengths']):.0f}, Max: {np.max(stats['claim_lengths']):.0f}"
    )
    print(f"    Std: {np.std(stats['claim_lengths']):.2f}")

    print(f"  Content length:")
    print(
        f"    Mean: {np.mean(stats['content_lengths']):.2f}, Median: {np.median(stats['content_lengths']):.2f}"
    )
    print(
        f"    Min: {np.min(stats['content_lengths']):.0f}, Max: {np.max(stats['content_lengths']):.0f}"
    )
    print(f"    Std: {np.std(stats['content_lengths']):.2f}")

    print(f"\nEvidence Statistics:")
    print(
        f"  Samples with evidence: {stats['has_evidence']}/{stats['total_samples']} ({(stats['has_evidence']/stats['total_samples']*100):.2f}%)"
    )
    print(f"  Mean evidence count: {np.mean(stats['evidence_counts']):.2f}")
    print(f"  Median evidence count: {np.median(stats['evidence_counts']):.2f}")
    print(f"  Max evidence count: {np.max(stats['evidence_counts']):.0f}")

    print(f"\nMultimedia Statistics:")
    print(
        f"  Samples with video: {stats['has_video']}/{stats['total_samples']} ({(stats['has_video']/stats['total_samples']*100):.2f}%)"
    )
    if stats["has_video"] > 0:
        print(
            f"    Video transcript length (mean): {np.mean(stats['video_transcript_lengths']):.2f} words"
        )
    print(
        f"  Samples with evidences: {stats['has_image']}/{stats['total_samples']} ({(stats['has_image']/stats['total_samples']*100):.2f}%)"
    )

    print(f"\n{'='*60}\n")


def plot_statistics(train_stats, val_stats, test_stats, output_dir="plots"):
    """Create visualization plots for dataset statistics."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Label distribution across splits
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, stats in enumerate([train_stats, val_stats, test_stats]):
        labels = list(stats["label_distribution"].keys())
        counts = list(stats["label_distribution"].values())
        axes[idx].bar(labels, counts, color=["green", "gray", "red"])
        axes[idx].set_title(f"{stats['split']} - Label Distribution")
        axes[idx].set_ylabel("Count")
        for i, v in enumerate(counts):
            axes[idx].text(i, v + 5, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "label_distribution.png"), dpi=100, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Text length distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (ax, stats) in enumerate(zip(axes, [train_stats, val_stats, test_stats])):
        ax.hist(stats["claim_lengths"], bins=30, alpha=0.7, label="Claim", color="blue")
        ax.set_title(f"{stats['split']} - Claim Length Distribution")
        ax.set_xlabel("Words")
        ax.set_ylabel("Count")
        ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "claim_length_distribution.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Evidence count distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (ax, stats) in enumerate(zip(axes, [train_stats, val_stats, test_stats])):
        evidence_counts = Counter(stats["evidence_counts"])
        counts_sorted = sorted(evidence_counts.items())
        labels = [str(c[0]) for c in counts_sorted]
        counts = [c[1] for c in counts_sorted]
        ax.bar(labels, counts, color="orange")
        ax.set_title(f"{stats['split']} - Evidence Count Distribution")
        ax.set_xlabel("Number of Evidence")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "evidence_count_distribution.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 4: Multimedia availability
    splits_names = [train_stats["split"], val_stats["split"], test_stats["split"]]
    video_pcts = [
        (train_stats["has_video"] / train_stats["total_samples"]) * 100,
        (val_stats["has_video"] / val_stats["total_samples"]) * 100,
        (test_stats["has_video"] / test_stats["total_samples"]) * 100,
    ]
    image_pcts = [
        (train_stats["has_image"] / train_stats["total_samples"]) * 100,
        (val_stats["has_image"] / val_stats["total_samples"]) * 100,
        (test_stats["has_image"] / test_stats["total_samples"]) * 100,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(splits_names))
    width = 0.35
    ax.bar(x - width / 2, video_pcts, width, label="Video", color="skyblue")
    ax.bar(x + width / 2, image_pcts, width, label="Images", color="lightcoral")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Multimedia Availability Across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(splits_names)
    ax.legend()
    ax.set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "multimedia_availability.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-dataloader", action="store_true", help="Test dataloader creation"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--statistics", action="store_true", help="Compute and print dataset statistics"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate visualization plots"
    )
    args = parser.parse_args()

    if args.statistics or args.plot:
        print("Loading dataset...")
        train_data, val_data, test_data = get_dataset(DATA_PATH)

        if args.statistics:
            train_stats = compute_dataset_statistics(train_data, "train_val")
            val_stats = compute_dataset_statistics(val_data, "train_val")
            test_stats = compute_dataset_statistics(test_data, "test")

            print_statistics(train_stats)
            print_statistics(val_stats)
            print_statistics(test_stats)

        if args.plot:
            train_stats = compute_dataset_statistics(train_data, "train_val")
            val_stats = compute_dataset_statistics(val_data, "train_val")
            test_stats = compute_dataset_statistics(test_data, "test")
            plot_statistics(train_stats, val_stats, test_stats)

    elif args.test_dataloader:
        print("Testing dataloader creation...")
        train_loader, val_loader, test_loader = create_dataloaders(
            path=DATA_PATH,
            batch_size=args.batch_size,
            limit_samples=8,
        )

        print("\nTesting one batch from train_loader:")
        for batch in train_loader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch size: {len(batch['claim'])}")
            print(f"Claims: {batch['claim'][:2]}")
            print(f"Ratings: {batch['rating'][:2]}")
            print(f"Labels shape: {batch['label'].shape}")
            print(
                f"Text evidence count (first sample): {len(batch['text_evidence'][0])}"
            )
            print(
                f"Video transcript length (first sample): {len(batch['video_transcript'][0])}"
            )
            break
    else:
        print("Usage: python true_dataset.py --statistics")
        print("       python true_dataset.py --plot")
        print("       python true_dataset.py --test-dataloader")
