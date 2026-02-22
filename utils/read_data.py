from cgi import test
import json
from multiprocessing import process
import time
import pandas as pd
import cv2
import glob
from tqdm import tqdm, trange
import torch
import numpy as np
import math
import random
import re
from torch.utils.data import DataLoader
from pathlib import Path
from rag import RAGEnhancedDataset, RAGEvidenceRetriever

DATA_PATH = "data/mocheg"


def read_text_corpus(path):
    train = path + "/train/Corpus2.csv"
    dev = path + "/val/Corpus2.csv"
    test = path + "/test/Corpus2.csv"

    train_data = pd.read_csv(train, low_memory=False)
    val_data = pd.read_csv(dev, low_memory=False)
    test_data = pd.read_csv(test, low_memory=False)

    return (train_data, val_data, test_data)


def read_image(path):
    # imdir = 'path/to/files/'
    ext = ["jpg", "jpeg", "png"]

    files = []
    images = []
    names = []
    claim = []
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            files.append(img)

    for f in files:
        names.append(f.split("/")[-1])
        claim.append(int(f.split("/")[-1].split("-")[0]))
        images.append(cv2.imread(f))

    # images = [cv2.imread(file) for file in files]
    #
    return pd.DataFrame({"claim_id": claim, "id": names, "image": images})


def read_image_path_only(path):
    ext = ["jpg", "jpeg", "png"]

    files = []
    images = []
    names = []
    claim = []
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            files.append(img)

    for f in files:
        names.append(f.split("\\")[-1])
        claim.append(int(f.split("\\")[-1].split("-")[0]))
        images.append(f)

    # images = [cv2.imread(file) for file in files]
    #
    return pd.DataFrame({"claim_id": claim, "id": names, "image": images})


def read_images_corpus(path):
    train = path + "/train/images"
    dev = path + "/val/images"
    test = path + "/test/images"

    train_images = read_image_path_only(train)
    dev_images = read_image_path_only(dev)
    test_images = read_image_path_only(test)

    return (train_images, dev_images, test_images)


def retrieve_data_for_verification(train_text, train_images):
    claim_ids = train_text["claim_id"].values
    claim_ids = list(set(claim_ids))

    claim_data = []
    for claim_id in tqdm(claim_ids):
        df = train_text.loc[(train_text.claim_id == claim_id)]
        text_evidences = df["Evidence"].values
        image_evidences = train_images.loc[(train_images.claim_id == claim_id)][
            "image"
        ].values

        claim_object = (
            df["Claim"].values[0],
            text_evidences,
            image_evidences,
            df["cleaned_truthfulness"].values[0],
            claim_id,
        )
        claim_data.append(claim_object)

    return claim_data


def get_dataset(path):
    train_text, dev_text, test_text = read_text_corpus(path)
    train_image, dev_image, test_image = read_images_corpus(path)

    train_claim = retrieve_data_for_verification(train_text, train_image)
    val_claim = retrieve_data_for_verification(dev_text, dev_image)
    test_claim = retrieve_data_for_verification(test_text, test_image)

    return train_claim, val_claim, test_claim


def one_hot(a, num_classes):
    v = np.zeros(num_classes, dtype=int)
    v[a] = 1
    return v


def clean_data(text):
    if str(text) == "nan":
        return text
    text = re.sub("(<p>|</p>|@)+", "", text)
    return text.strip()


def encode_one_sample(sample, rag_evidence=None):
    claim = sample[0]
    text_evidence = sample[1]
    image_evidence = sample[2]
    label = sample[3]
    claim_id = sample[4]

    label2idx = {"refuted": 2, "NEI": 1, "supported": 0}

    cleaned_text_evidence = [clean_data(t) for t in text_evidence]

    encoded_sample = {
        "claim_id": claim_id,
        "claim": claim,
        "label": torch.tensor(one_hot(label2idx[label], 3), dtype=float),
        "text_evidence": cleaned_text_evidence,
        "external_evidence": (rag_evidence if rag_evidence else ""),
        "image_evidence": image_evidence.tolist(),
    }

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data, rag_cache_path=None, use_rag=False):
        self._data = claim_verification_data

        self.use_rag = use_rag
        self.rag_cache = {}

        if use_rag and rag_cache_path and Path(rag_cache_path).exists():
            print(f"Loading RAG cache from {rag_cache_path}")
            with open(rag_cache_path, "r", encoding="utf-8") as f:
                self.rag_cache = json.load(f)
            print(f"Loaded {len(self.rag_cache)} RAG evidence entries")

        self._encoded = []
        for d in self._data:
            claim = d[0]
            rag_evidence = self.rag_cache.get(claim, None) if self.use_rag else None
            self._encoded.append(encode_one_sample(d, rag_evidence))
            # self._encoded.append(encode_one_sample(d))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def collate_claim_verification(batch):
    claim_ids = [item["claim_id"] for item in batch]
    claims = [item["claim"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    text_evidences = [item["text_evidence"] for item in batch]
    image_evidences = [item["image_evidence"] for item in batch]
    external_evidences = [item.get("external_evidence", "") for item in batch]

    return {
        "claim_id": claim_ids,
        "claim": claims,
        "label": labels,
        "text_evidence": text_evidences,
        "image_evidence": image_evidences,
        "external_evidence": external_evidences,
    }


def create_dataloaders(
    path=DATA_PATH,
    batch_size=32,
    shuffle_train=True,
    num_workers=0,
    pin_memory=False,
    use_rag=False,
    rag_cache_dir=None,
    limit_samples=None,
):
    train_claim, val_claim, test_claim = get_dataset(path)

    train_rag_cache = None
    val_rag_cache = None
    test_rag_cache = None

    if use_rag and rag_cache_dir:
        rag_cache_path = Path(rag_cache_dir)
        train_rag_cache = str(rag_cache_path / "train_rag_cache.json")
        val_rag_cache = str(rag_cache_path / "val_rag_cache.json")
        test_rag_cache = str(rag_cache_path / "test_rag_cache.json")

        print(f"\nRAG Evidence Integration: {'ENABLED' if use_rag else 'DISABLED'}")
        if use_rag:
            print(f"RAG cache directory: {rag_cache_dir}")

    train_dataset = ClaimVerificationDataset(
        train_claim, rag_cache_path=train_rag_cache, use_rag=use_rag
    )
    val_dataset = ClaimVerificationDataset(
        val_claim, rag_cache_path=val_rag_cache, use_rag=use_rag
    )
    test_dataset = ClaimVerificationDataset(
        test_claim, rag_cache_path=test_rag_cache, use_rag=use_rag
    )

    # train_dataset = train_dataset[:8]
    # val_dataset = val_dataset[:8]
    # test_dataset = test_dataset[:8]

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


def precompute_rag_evidence(
    data_path: str = DATA_PATH,
    output_dir: str = "cache/rag",
    search_engine: str = "duckduckgo",
    top_k_articles: int = 5,
    max_search_results: int = 10,
    extract_full_content: bool = True,
):
    """
    Pre-compute RAG evidence for all claims in the dataset.
    This should be run ONCE before training to cache all evidence.
    """

    print("=" * 80)
    print("PRE-COMPUTING RAG EVIDENCE FOR MOCHEG DATASET")
    print("=" * 80)

    # Initialize RAG retriever
    rag_retriever = RAGEvidenceRetriever(
        search_engine=search_engine,
        max_search_results=max_search_results,
        top_k_articles=top_k_articles,
        extract_full_content=extract_full_content,
        cache_dir=f"{output_dir}/search_cache",
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_claim, val_claim, test_claim = get_dataset(data_path)

    splits = {"train": train_claim, "val": val_claim, "test": test_claim}

    for split_name, claim_data in splits.items():
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} split ({len(claim_data)} claims)")
        print(f"{'='*80}")

        cache_file = output_path / f"{split_name}_rag_cache.json"

        # Load existing cache if available
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                rag_cache = json.load(f)
            print(f"Loaded existing cache with {len(rag_cache)} entries")
        else:
            rag_cache = {}

        # Process each claim
        for i, sample in enumerate(tqdm(claim_data, desc=f"Retrieving {split_name}")):
            claim = sample[0]  # Claim text
            claim_id = sample[4]  # Claim ID

            if claim in rag_cache:
                continue

            try:
                result = rag_retriever.retrieve_evidence(claim)
                rag_cache[claim] = result["combined_evidence"]

                if (i + 1) % 10 == 0:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(rag_cache, f, indent=2, ensure_ascii=False)
                    print(f"\n✓ Checkpoint saved: {len(rag_cache)} claims processed")

                time.sleep(1.5)

            except Exception as e:
                print(f"\n✗ Error processing claim {claim_id}: {e}")
                rag_cache[claim] = ""  # Empty evidence on error
                continue

        # Final save
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(rag_cache, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Completed {split_name} split")
        print(f"  Total claims: {len(claim_data)}")
        print(f"  Cached evidence: {len(rag_cache)}")
        print(f"  Saved to: {cache_file}")

    print("\n" + "=" * 80)
    print("RAG EVIDENCE PRE-COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nCache files saved to: {output_path}")
    print("\nTo use RAG evidence in training:")
    print("  train_loader, val_loader, test_loader = create_dataloaders(")
    print("      use_rag=True,")
    print(f"      rag_cache_dir='{output_dir}'")
    print("  )")


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precompute-rag", action="store_true", help="Pre-compute RAG evidence"
    )
    parser.add_argument(
        "--test-dataloader", action="store_true", help="Test dataloader creation"
    )
    parser.add_argument(
        "--use-rag", action="store_true", help="Use RAG evidence (for testing)"
    )
    parser.add_argument(
        "--rag-cache-dir", type=str, default="cache/rag", help="RAG cache directory"
    )
    args = parser.parse_args()

    if args.precompute_rag:
        # Pre-compute RAG evidence
        precompute_rag_evidence(
            data_path=DATA_PATH,
            output_dir=args.rag_cache_dir,
            search_engine="duckduckgo",
            top_k_articles=5,
            extract_full_content=True,
        )

    elif args.test_dataloader:
        # Test dataloader creation
        print("Testing dataloader creation...")
        train_loader, val_loader, test_loader = create_dataloaders(
            path=DATA_PATH,
            batch_size=4,
            use_rag=args.use_rag,
            rag_cache_dir=args.rag_cache_dir if args.use_rag else None,
            limit_samples=8,  # Test with small dataset
        )

        # Test one batch
        print("\nTesting one batch from train_loader:")
        for batch in train_loader:
            print(f"\nBatch keys: {batch.keys()}")
            print(f"Batch size: {len(batch['claim'])}")
            print(f"Claims: {batch['claim'][:2]}")
            print(f"Labels shape: {batch['label'].shape}")
            print(f"Text evidence (first sample): {batch['text_evidence'][0][:200]}...")
            print(f"Image evidence (first sample): {batch['image_evidence'][0][:2]}")
            break

    else:
        print("Usage:")
        print(
            "  python read_data.py --precompute-rag                    # Pre-compute RAG evidence"
        )
        print(
            "  python read_data.py --test-dataloader                   # Test without RAG"
        )
        print(
            "  python read_data.py --test-dataloader --use-rag         # Test with RAG"
        )
