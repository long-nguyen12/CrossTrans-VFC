import os
os.environ.setdefault("DECORD_LOG_LEVEL", "3")
os.environ.setdefault("FFMPEG_LOG_LEVEL", "error")
from pathlib import Path
import torch
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from decord import VideoReader, cpu
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, Dict
import argparse

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
    except Exception:
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
):
    try:
        vr = _call_quiet_stderr(
            VideoReader, video_path, num_threads=1, ctx=cpu(0), fault_tol=32
        )
    except TypeError:
        print(f"Warning: 'fault_tol' not supported in this version of decord. Proceeding without it.")
        vr = _call_quiet_stderr(VideoReader, video_path, num_threads=1, ctx=cpu(0))
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

    # Bound runtime on long videos by sampling clip anchors uniformly.
    if max_clips is not None and max_clips > 0 and start_indices.size > max_clips:
        sampled_idx = np.linspace(
            0, start_indices.size - 1, num=max_clips, dtype=np.int64
        )
        start_indices = start_indices[sampled_idx]

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
                elif isinstance(token_features, (tuple, list)) and len(token_features) > 0:
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
        print(f"Warning: No valid clips extracted from video {video_path}. Returning zero features.")
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.zeros(1, 1, hidden_size)

    video_sequence = torch.cat(all_features, dim=0).unsqueeze(0)
    return video_sequence


# class VideoMAEExtractor:
#     """Wrapper for VideoMAE feature extraction."""

#     def __init__(self, model_name: str = "MCG-NJU/videomae-base", device: str = "cuda"):
#         self.device = device if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")

#         # Load model and processor
#         config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
#         self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(
#             model_name, config=config, trust_remote_code=True
#         ).to(self.device)

#         self.model.eval()
#         self.model.requires_grad_(False)

#         # Get default clip length from config
#         self.default_clip_len = getattr(config, "num_frames", 16)
#         print(f"Model loaded: {model_name}")
#         print(f"Default clip length: {self.default_clip_len} frames")

#     def sample_frame_indices(
#         self,
#         clip_len: int,
#         frame_sample_rate: int,
#         seg_len: int,
#         random_sample: bool = False,
#     ) -> np.ndarray:
#         converted_len = int(clip_len * frame_sample_rate)

#         if seg_len <= 0 or converted_len <= 0:
#             raise ValueError("Invalid seg_len or converted_len")

#         # If converted length exceeds video length, sample uniformly
#         if converted_len >= seg_len:
#             index = np.linspace(0, seg_len, num=clip_len, endpoint=False)
#             return np.clip(index, 0, seg_len - 1).astype(np.int64)

#         # Random or center sampling
#         if random_sample:
#             end_idx = np.random.randint(converted_len, seg_len)
#             start_idx = end_idx - converted_len
#         else:
#             # Center sampling (deterministic)
#             start_idx = (seg_len - converted_len) // 2
#             end_idx = start_idx + converted_len

#         index = np.linspace(start_idx, end_idx, num=clip_len, endpoint=False)
#         return np.clip(index, start_idx, end_idx - 1).astype(np.int64)

#     def extract_single_clip(
#         self,
#         video_path: str,
#         clip_len: Optional[int] = None,
#         frame_sample_rate: int = 4,
#         random_sample: bool = False,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if clip_len is None:
#             clip_len = self.default_clip_len

#         vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
#         vr.seek(0)
#         seg_len = len(vr)

#         indices = self.sample_frame_indices(
#             clip_len, frame_sample_rate, seg_len, random_sample
#         )

#         buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
#         frames = [buffer[i] for i in range(buffer.shape[0])]

#         inputs = self.processor(frames, return_tensors="pt")

#         inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         if hasattr(outputs, "last_hidden_state"):
#             token_features = outputs.last_hidden_state  # [B, N, D]
#         elif isinstance(outputs, (tuple, list)):
#             token_features = outputs[0]
#         else:
#             token_features = outputs

#         # Average pooling over tokens for video-level feature
#         video_features = token_features.mean(dim=1)  # [B, D]

#         return video_features, token_features

#     def extract_multi_clip(
#         self,
#         video_path: str,
#         num_clips: int = 8,
#         clip_len: Optional[int] = None,
#         aggregation: str = "mean",
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if clip_len is None:
#             clip_len = self.default_clip_len

#         vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
#         seg_len = len(vr)

#         # Calculate clip span
#         clip_span = max(seg_len // num_clips, clip_len)

#         all_video_feats = []
#         all_token_feats = []

#         for i in range(num_clips):
#             # Calculate start/end for this clip
#             start = min(i * clip_span, max(0, seg_len - clip_len))
#             end = min(start + clip_span, seg_len)

#             # Sample frames for this clip
#             index = np.linspace(
#                 start, max(start + 1, end), num=clip_len, endpoint=False
#             )
#             index = np.clip(index, 0, seg_len - 1).astype(np.int64)

#             # Get frames
#             buffer = vr.get_batch(index).asnumpy()
#             frames = [buffer[j] for j in range(buffer.shape[0])]

#             # Process frames
#             inputs = self.processor(frames, return_tensors="pt")
#             inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             # Extract features
#             with torch.no_grad():
#                 outputs = self.model(**inputs)

#             token_features = (
#                 outputs.last_hidden_state
#                 if hasattr(outputs, "last_hidden_state")
#                 else outputs[0]
#             )
#             video_features = token_features.mean(dim=1)  # [1, D]

#             all_video_feats.append(video_features)
#             all_token_feats.append(token_features)

#         # Aggregate video features
#         if aggregation == "mean":
#             video_features = torch.cat(all_video_feats, dim=0).mean(dim=0, keepdim=True)
#         elif aggregation == "concat":
#             video_features = torch.cat(all_video_feats, dim=1)  # [1, num_clips*D]
#         else:
#             raise ValueError(f"Unknown aggregation: {aggregation}")

#         # Concatenate all token features
#         token_features = torch.cat(all_token_feats, dim=1)  # [1, num_clips*N, D]

#         return video_features, token_features

#     def extract_from_directory(
#         self,
#         video_dir: str,
#         output_dir: str,
#         video_extension: str = ".mp4",
#         multi_clip: bool = False,
#         num_clips: int = 8,
#         overwrite: bool = False,
#     ) -> None:
#         video_dir = Path(video_dir)
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         # Get all video files
#         video_files = sorted(video_dir.glob(f"*{video_extension}"))
#         print(f"Found {len(video_files)} videos in {video_dir}")

#         # Process each video
#         for video_path in tqdm(video_files, desc="Extracting features"):
#             video_name = video_path.stem
#             save_path = output_dir / f"{video_name}.pt"

#             # Skip if already exists
#             if save_path.exists() and not overwrite:
#                 continue

#             try:
#                 # Extract features
#                 if multi_clip:
#                     video_features, token_features = self.extract_multi_clip(
#                         str(video_path), num_clips=num_clips
#                     )
#                 else:
#                     video_features, token_features = self.extract_single_clip(
#                         str(video_path)
#                     )

#                 # Save features
#                 torch.save(
#                     {
#                         "video_features": video_features.cpu(),
#                         "token_features": token_features.cpu(),
#                         "video_path": str(video_path),
#                     },
#                     save_path,
#                 )

#             except Exception as e:
#                 print(f"Error processing {video_name}: {e}")
#                 continue

#         print(f"Features saved to {output_dir}")

#     def extract_long_video(
#         self,
#         video_path: str,
#         stride_seconds: float = 2.0,
#         clip_len: int = 16,
#     ):
#         vr = VideoReader(video_path, num_threads=2, ctx=cpu(0))
#         fps = vr.get_avg_fps()
#         total_frames = len(vr)

#         stride_frames = int(stride_seconds * fps)

#         all_features = []

#         for start_idx in range(0, total_frames - clip_len, stride_frames):
#             indices = np.arange(start_idx, start_idx + clip_len)

#             buffer = vr.get_batch(indices).asnumpy()
#             frames = [buffer[i] for i in range(buffer.shape[0])]

#             inputs = self.processor(frames, return_tensors="pt")
#             if "videomaev2" in self.model.config.model_type.lower():
#                 inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)

#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 outputs = self.model(**inputs)

#                 if hasattr(outputs, "last_hidden_state"):
#                     token_features = outputs.last_hidden_state
#                 elif isinstance(outputs, (tuple, list)):
#                     token_features = outputs[0]
#                 else:
#                     token_features = outputs

#                 if token_features.dim() == 3:
#                     feat = token_features.mean(dim=1).cpu()
#                 else:
#                     feat = token_features.cpu()
#                 all_features.append(feat)

#         video_sequence = torch.cat(all_features, dim=0).unsqueeze(0)
#         return video_sequence


# def main():
#     parser = argparse.ArgumentParser(
#         description="Extract VideoMAE features from videos"
#     )
#     parser.add_argument(
#         "--video_dir", type=str, required=True, help="Directory containing videos"
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         required=True,
#         help="Directory to save extracted features",
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="MCG-NJU/videomae-base",
#         help="VideoMAE model name from HuggingFace",
#     )
#     parser.add_argument(
#         "--video_extension", type=str, default=".mp4", help="Video file extension"
#     )
#     parser.add_argument(
#         "--multi_clip",
#         action="store_true",
#         help="Use multi-clip extraction for long videos",
#     )
#     parser.add_argument(
#         "--num_clips",
#         type=int,
#         default=8,
#         help="Number of clips for multi-clip extraction",
#     )
#     parser.add_argument(
#         "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
#     )
#     parser.add_argument(
#         "--overwrite", action="store_true", help="Overwrite existing features"
#     )

#     args = parser.parse_args()

#     # Initialize extractor
#     extractor = VideoMAEExtractor(model_name=args.model_name, device=args.device)

#     # Extract features
#     extractor.extract_from_directory(
#         video_dir=args.video_dir,
#         output_dir=args.output_dir,
#         video_extension=args.video_extension,
#         multi_clip=args.multi_clip,
#         num_clips=args.num_clips,
#         overwrite=args.overwrite,
#     )


# if __name__ == "__main__":
#     extractor = VideoMAEExtractor(model_name="OpenGVLab/VideoMAEv2-Base")

#     video_path = "./1942500.mp4"
#     video_features = extractor.extract_long_video(video_path)
#     print(f"Video features shape: {video_features.shape}")  # [1, 768]
#     # print(f"Token features shape: {token_features.shape}")  # [1, N, 768]

#     # Batch extraction example
#     # extractor.extract_from_directory(
#     #     video_dir="path/to/videos",
#     #     output_dir="path/to/features",
#     #     multi_clip=True,
#     #     num_clips=8
#     # )
