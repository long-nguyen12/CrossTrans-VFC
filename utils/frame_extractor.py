import logging
import os
from pathlib import Path
from config import DATASET_CONFIG, VIDEO_DESCRIPTOR_CONFIG
import argparse

import cv2
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from sklearn.cluster import SpectralClustering
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def reorder_and_rename_images(directory_path):
    directory = Path(directory_path)
    images = sorted(directory.glob("*.jpeg"), key=lambda p: p.stat().st_mtime)
    if not images:
        logging.info("No JPEG images found to rename in %s", directory_path)
        return

    # Use temporary names to avoid collisions when target names already exist.
    tmp_paths = []
    for idx, image_path in enumerate(images, start=1):
        tmp_path = image_path.with_name(f"__tmp_{idx:06d}.jpeg")
        image_path.rename(tmp_path)
        tmp_paths.append(tmp_path)

    for idx, tmp_path in enumerate(tmp_paths, start=1):
        tmp_path.rename(directory / f"{idx}.jpeg")

    logging.info("Images have been renamed successfully.")


def _load_clip_vision(model_name):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _read_frames_at_indices(cap, indices):
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def _select_representative_frame(frames, processor, model, device):
    if not frames:
        return None
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings


def _select_representative_frame_spectral(frames, embeddings, n_clusters):
    if not frames:
        return None
    if embeddings.shape[0] == 1:
        return frames[0]

    n_clusters = max(2, min(int(n_clusters), int(embeddings.shape[0])))
    n_samples = int(embeddings.shape[0])
    n_neighbors = max(1, min(10, n_samples - 1))
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=0,
    ).fit_predict(embeddings.cpu().numpy())

    # Choose the medoid of the largest cluster.
    largest_label = max(set(labels), key=lambda l: (labels == l).sum())
    idxs = [i for i, l in enumerate(labels) if l == largest_label]
    cluster_emb = embeddings[idxs]
    centroid = cluster_emb.mean(dim=0, keepdim=True)
    distances = torch.cdist(cluster_emb, centroid).squeeze(1)
    best_local = int(torch.argmin(distances).item())
    return frames[idxs[best_local]]


def clip_chunk_keyframes_extraction(
    model,
    processor,
    video_file_path,
    chunk_count=10,
    samples_per_chunk=8,
    spectral_clusters=2,
    output_dir=None,
    device=None,
):
    video_path = Path(video_file_path)
    if output_dir is None:
        target_path = video_path.parent / video_path.stem
    else:
        target_path = Path(output_dir) / video_path.stem

    if target_path.exists() and len(list(target_path.glob("*.jpeg"))) >= chunk_count:
        logging.info("Keyframes already extracted and present in %s", target_path)
        return str(target_path)

    cap = cv2.VideoCapture(str(video_file_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_file_path}")

    chunk_count = min(chunk_count, total_frames)
    chunk_size = total_frames / chunk_count

    print(f"Using device: {device}")

    target_path.mkdir(parents=True, exist_ok=True)
    saved = 0

    for chunk_idx in range(chunk_count):
        start = int(chunk_idx * chunk_size)
        end = int(min(total_frames, (chunk_idx + 1) * chunk_size))
        if end <= start:
            continue

        if samples_per_chunk <= 1 or (end - start) <= 1:
            indices = [start]
        else:
            step = max(1, (end - start) // samples_per_chunk)
            indices = list(range(start, end, step))[:samples_per_chunk]

        frames = _read_frames_at_indices(cap, indices)
        start = time.time()
        embeddings = _select_representative_frame(frames, processor, model, device)
        end = time.time()
        print("Encoded time is: ", end - start)
        if embeddings is None:
            continue
        start = time.time()
        best_frame = _select_representative_frame_spectral(
            frames, embeddings, spectral_clusters
        )
        end = time.time()
        print("Clustering time is: ", end - start)
        if best_frame is None:
            continue

        saved += 1
        out_path = target_path / f"{saved}.jpeg"
        out_bgr = cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)

    cap.release()

    reorder_and_rename_images(str(target_path))
    logging.info(
        "video %s: CLIP chunk keyframes extracted successfully", video_path.stem
    )
    return str(target_path)


def process_folder_videos(args):
    dataset_root = Path(f"{DATASET_CONFIG['root_dir']}").resolve()
    # Writable output root
    output_root = Path(f"{DATASET_CONFIG['root_dir']}").resolve()

    test_annotation = dataset_root / DATASET_CONFIG["annotation"][args.split]
    test_video_dir = dataset_root / DATASET_CONFIG["video_dir"][args.split]

    test_output_dir = output_root / DATASET_CONFIG["output_dir"][args.split]

    test_output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_annotation, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = VIDEO_DESCRIPTOR_CONFIG.get(
        "clip_model_name", "openai/clip-vit-base-patch32"
    )
    processor, model = _load_clip_vision(model_name)
    model.to(device)

    for i, video_id in enumerate(video_ids):
        try:
            logging.info(f"Processing video {i + 1}/{len(video_ids)}: {video_id}")

            video_file = None
            for ext in [".mp4", ".mkv"]:
                potential_file = os.path.join(test_video_dir, f"{video_id}{ext}")
                if os.path.exists(potential_file):
                    video_file = potential_file
                    break

            if not video_file:
                logging.warning(f"No video file found for ID: {video_id}")
                continue

            data_folder = os.path.join(test_video_dir, str(video_id))

            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            logging.info(f"Extracting keyframes for video: {video_id}")
            try:
                chunk_count = VIDEO_DESCRIPTOR_CONFIG.get("chunk_count", 10)
                samples_per_chunk = VIDEO_DESCRIPTOR_CONFIG.get(
                    "clip_samples_per_chunk", 8
                )
                spectral_clusters = VIDEO_DESCRIPTOR_CONFIG.get(
                    "spectral_clusters_per_chunk", 2
                )

                _ = clip_chunk_keyframes_extraction(
                    model,
                    processor,
                    video_file,
                    chunk_count=chunk_count,
                    samples_per_chunk=samples_per_chunk,
                    spectral_clusters=spectral_clusters,
                    output_dir=test_output_dir,
                    device=device,
                )
                logging.info(f"Keyframes extracted successfully for video: {video_id}")
            except Exception as e:
                logging.error(
                    f"Failed to extract keyframes for video {video_id}: {str(e)}"
                )
                continue

        except Exception as e:
            logging.error(f"Error processing video {video_id}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/TRUE_Dataset",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train_val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/video_cache",
        help="Dir with pre-extracted video feature .pt files",
    )
    args = parser.parse_args()
    process_folder_videos(args)
