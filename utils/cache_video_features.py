import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor, AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.vid_extractor import extract_video_folder


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract VideoMAE features for all dataset videos."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/TRUE_Dataset",
        help="Root directory of TRUE_Dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/video_cache",
        help="Directory to save cached .pt feature tensors",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="MCG-NJU/videomae-base",
        help="VideoMAE model id",
    )
    parser.add_argument("--stride-frames", type=int, default=64)
    parser.add_argument("--max-clips", type=int, default=24)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for decord VideoReader",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos for testing (None = all)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = [
        data_root / "train_val_video",
        data_root / "test_video",
    ]

    # Pre-check to find which videos actually need processing
    videos_to_process = []
    skipped = 0
    for vdir in video_dirs:
        if not vdir.exists():
            continue
        for vf in vdir.glob("*.mp4"):
            video_id = vf.stem
            cache_path = cache_dir / f"{video_id}.pt"
            if cache_path.exists():
                skipped += 1
                pass  # "if founded, pass it"
            else:
                videos_to_process.append(vf)

    print(
        f"Found {skipped} videos already cached. {len(videos_to_process)} videos need processing."
    )

    if not videos_to_process:
        print("All videos are already cached! No need to load the model.")
        return

    # Load model only if there's work to do
    print(f"Loading VideoMAE model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    model.requires_grad_(False)

    from tqdm import tqdm
    from utils.vid_extractor import extract_long_video

    for vf in tqdm(videos_to_process, desc="Extracting video features"):
        video_id = vf.stem
        cache_path = cache_dir / f"{video_id}.pt"

        try:
            feats = extract_long_video(
                model,
                processor,
                str(vf),
                stride_seconds=2.0,
                stride_frames=args.stride_frames,
                clip_len=None,
                max_clips=args.max_clips,
                num_threads=args.num_threads,
            )  # [1, T, D]
            feats = feats.squeeze(0)  # [T, D]
            torch.save(feats, cache_path)
        except Exception as e:
            print(f"Error caching {vf.name}: {e}")

    print(f"\n✓ Done processing {len(videos_to_process)} videos.")


if __name__ == "__main__":
    main()
