import os

os.environ.setdefault("DECORD_LOG_LEVEL", "3")
os.environ.setdefault("FFMPEG_LOG_LEVEL", "error")
from pathlib import Path
import torch
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from decord import VideoReader, cpu
from tqdm import tqdm
import numpy as np
from typing import Optional
import argparse
import time


def _call_quiet_stderr(fn, *args, **kwargs):
    """Call a function while temporarily silencing native stderr (POSIX)."""
    if os.name != "posix":
        return fn(*args, **kwargs)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        return fn(*args, **kwargs)
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


def _safe_read_clip(vr: VideoReader, indices: np.ndarray) -> np.ndarray:
    """Read a clip robustly; fall back to per-frame decode on batch errors."""
    try:
        return np.ascontiguousarray(_call_quiet_stderr(vr.get_batch, indices).asnumpy())
    except Exception as e:
        print(f"Batch read failed with error: {e}")
        print(
            f"Batch read failed for frames {indices}. Falling back to per-frame decode."
        )
        decoded_frames = []
        for idx in indices.tolist():
            try:
                frame = _call_quiet_stderr(vr.__getitem__, int(idx))
                decoded_frames.append(np.ascontiguousarray(frame.asnumpy()))
            except Exception:
                continue

        if not decoded_frames:
            raise

        while len(decoded_frames) < len(indices):
            decoded_frames.append(decoded_frames[-1])

        return np.stack(decoded_frames[: len(indices)], axis=0)


def extract_long_video(
    model,
    processor,
    video_path,
    stride_seconds: float = 2.0,
    stride_frames: Optional[int] = 64,
    clip_len: Optional[int] = None,
    max_clips: int = 24,
    num_threads: int = 4,
):
    try:
        vr = _call_quiet_stderr(
            VideoReader,
            video_path,
            num_threads=1,
            ctx=cpu(0),
            fault_tol=-1,
            width=224,
            height=224,
        )
    except TypeError:
        vr = _call_quiet_stderr(
            VideoReader,
            video_path,
            num_threads=1,
            ctx=cpu(0),
            width=224,
            height=224,
        )
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    if clip_len is None:
        if hasattr(model, "get_vision_features"):
            # Match test.py / VJEPA2 default temporal window.
            clip_len = 64
        else:
            clip_len = getattr(model.config, "num_frames", 16)
    clip_len = max(1, int(clip_len))

    if total_frames <= 0:
        print(f"Warning: Video {video_path} has no frames. Returning zero features.")
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.zeros(1, 1, hidden_size)

    if stride_frames is None:
        stride_frames = max(1, int(stride_seconds * fps))
    stride_frames = max(1, stride_frames)

    all_features = []

    max_start = max(total_frames - clip_len, 0)
    start_indices = np.arange(0, max_start + 1, stride_frames, dtype=np.int64)
    if start_indices.size == 0:
        start_indices = np.array([0], dtype=np.int64)

    if max_clips is not None and max_clips > 0 and start_indices.size > max_clips:
        sampled_idx = np.linspace(
            0, start_indices.size - 1, num=max_clips, dtype=np.int64
        )
        start_indices = start_indices[sampled_idx]

    print(
        f"Extracting video features from {video_path} with {len(start_indices)} clips..."
    )
    for start_idx in start_indices.tolist():
        indices = np.arange(start_idx, start_idx + clip_len)
        indices = np.clip(indices, 0, max(total_frames - 1, 0))

        try:
            buffer = _safe_read_clip(vr, indices)
        except Exception:
            continue
        frames = [np.ascontiguousarray(frame) for frame in buffer]

        inputs = processor(frames, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
        if "videomaev2" in model_type:
            inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            if hasattr(model, "get_vision_features"):
                token_features = model.get_vision_features(**inputs)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    token_features = outputs.last_hidden_state
                elif isinstance(outputs, (tuple, list)):
                    token_features = outputs[0]
                else:
                    token_features = outputs

            if not isinstance(token_features, torch.Tensor):
                if hasattr(token_features, "last_hidden_state"):
                    token_features = token_features.last_hidden_state
                elif isinstance(token_features, dict):
                    token_features = token_features.get(
                        "last_hidden_state", token_features.get("pooler_output")
                    )
                elif (
                    isinstance(token_features, (tuple, list))
                    and len(token_features) > 0
                ):
                    token_features = token_features[0]

            if not isinstance(token_features, torch.Tensor):
                raise TypeError(
                    f"Unsupported video feature output type: {type(token_features)}"
                )

            if token_features.dim() == 1:
                feat = token_features.unsqueeze(0)
            elif token_features.dim() == 2:
                feat = token_features
            else:
                feat = token_features.reshape(
                    token_features.shape[0], -1, token_features.shape[-1]
                ).mean(dim=1)

            all_features.append(feat.cpu())

    if not all_features:
        print(
            f"Warning: No valid clips extracted from video {video_path}. Returning zero features."
        )
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.zeros(1, 1, hidden_size)

    video_sequence = torch.cat(all_features, dim=0).unsqueeze(0)
    return video_sequence


def extract_video_folder(
    folder,
    model,
    processor,
    cache_dir,
    stride_seconds: float = 2.0,
    stride_frames: Optional[int] = 64,
    clip_len: Optional[int] = None,
    max_clips: int = 24,
    num_threads: int = 4,
):
    """Pre-extract video features for all .mp4 files in *folder* and save to *cache_dir*.

    Each video ``<id>.mp4`` produces a cache file ``<cache_dir>/<id>.pt`` containing
    a tensor of shape ``[T, D]`` (num_clips, hidden_dim). Videos that already have
    a cached file are skipped so the process is resumable.

    Returns
    -------
    dict[str, Path]
        Mapping from video id to the path of the cached ``.pt`` file.
    """
    folder = Path(folder)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(folder.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {folder}")
        return {}

    cached = {}
    skipped = 0

    for vf in tqdm(video_files, desc=f"Caching features from {folder.name}"):
        video_id = vf.stem
        cache_path = cache_dir / f"{video_id}.pt"

        if cache_path.exists():
            cached[video_id] = cache_path
            skipped += 1
            continue

        try:
            feats = extract_long_video(
                model,
                processor,
                str(vf),
                stride_seconds=stride_seconds,
                stride_frames=stride_frames,
                clip_len=clip_len,
                max_clips=max_clips,
                num_threads=num_threads,
            )  # [1, T, D]
            feats = feats.squeeze(0)  # [T, D]
            torch.save(feats, cache_path)
            cached[video_id] = cache_path
        except Exception as e:
            print(f"Error caching {vf.name}: {e}")

    print(f"Done: {len(cached)} cached, {skipped} skipped (already exist)")
    return cached


def main():
    parser = argparse.ArgumentParser(description="Test extract_long_video function")
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to video file",
        default="C:\\Users\\hnguyen\\Documents\\PhD\\Code\\TRUE-3MFact\\data\\TRUE_Dataset\\test_video\\2038206.mp4",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MCG-NJU/videomae-base",
        help="Model name or path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--stride_seconds", type=float, default=2.0, help="Stride in seconds"
    )
    parser.add_argument(
        "--stride_frames", type=int, default=64, help="Stride in frames"
    )
    parser.add_argument("--clip_len", type=int, default=None, help="Clip length")
    parser.add_argument(
        "--max_clips", type=int, default=24, help="Maximum number of clips"
    )
    args = parser.parse_args()

    device = (
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name, config=config, trust_remote_code=True
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    start = time.time()
    video_features = extract_long_video(
        model,
        processor,
        args.video_path,
        stride_seconds=args.stride_seconds,
        stride_frames=args.stride_frames,
        clip_len=args.clip_len,
        max_clips=args.max_clips,
        num_threads=4,
    )
    end = time.time()
    print(f"Extracted video features in {end - start:.2f} seconds")
    print(f"Extracted video features shape: {video_features.shape}")


if __name__ == "__main__":
    main()
