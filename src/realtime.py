"""Real-time sign recognition from a webcam.

Webcam frame -> MediaPipe Holistic (Tasks API, unflipped RGB) -> (543, 3) array
in the parquet layout -> ring buffer of the last --window frames ->
extract.preprocess() -> model -> smoothed word output.

Needs the MediaPipe Holistic Landmarker model file (holistic_landmarker.task),
passed with --holistic-model.

Example:
  python -m src.realtime --ckpt checkpoints/tcn_last.pt --mirror
  python -m src.realtime --video clip.mp4 --save annotated.mp4   # no webcam needed
Keys: q quit, c clear words.
"""
import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision

from src.extract import LH, POSE, RH, preprocess
from src.train import MODELS

SIGN_MAP = Path("data/raw/sign_to_prediction_index_map.json")
N_LANDMARKS = 543  # parquet layout: face 0-467, left hand 468-488, pose 489-521, right hand 522-542
DEFAULT_WINDOW = 22  # median raw clip length (2000 random train.csv files, seed 42)


def load_model(ckpt_path: Path) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MODELS[ckpt["args"]["model"]]()
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["epoch"]


def load_labels(path: Path = SIGN_MAP) -> dict[int, str]:
    return {i: s for s, i in json.load(open(path)).items()}


def result_to_frame(result) -> np.ndarray:
    """Holistic result -> (543, 3) float32, NaN where nothing was detected."""
    frame = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for rows, lms in ((LH, result.left_hand_landmarks), (POSE, result.pose_landmarks),
                      (RH, result.right_hand_landmarks)):
        if lms:
            frame[rows] = [(p.x, p.y, p.z) for p in lms]
    return frame


def has_hand(frame: np.ndarray) -> bool:
    return bool(np.isfinite(frame[LH]).any() or np.isfinite(frame[RH]).any())


def window_to_input(frames) -> torch.Tensor:
    """Sequence of (543, 3) frames -> (1, 64, 225) float32, same path as extract.py."""
    x = preprocess(np.stack(list(frames)).astype(np.float32))  # (64, 75, 3)
    return torch.from_numpy(x.reshape(1, len(x), -1).astype(np.float32))


class Smoother:
    """Emit a word when it wins the majority of the last n top-1 predictions."""

    def __init__(self, n: int, threshold: float):
        self.history = deque(maxlen=n)
        self.threshold = threshold
        self.last_word = None

    def reset(self) -> None:
        self.history.clear()
        self.last_word = None

    def update(self, cls: int, prob: float):
        self.history.append((cls, prob))
        if len(self.history) < self.history.maxlen:
            return None
        winner, votes = Counter(c for c, _ in self.history).most_common(1)[0]
        mean_prob = np.mean([p for c, p in self.history if c == winner])
        if votes > len(self.history) / 2 and mean_prob >= self.threshold \
                and winner != self.last_word:
            self.last_word = winner
            return winner
        return None


def draw_landmarks(img: np.ndarray, frame: np.ndarray, mirror: bool) -> None:
    h, w = img.shape[:2]
    for rows, color in ((LH, (0, 200, 0)), (POSE, (200, 200, 200)), (RH, (0, 120, 255))):
        for x, y, _ in frame[rows]:
            if np.isfinite(x) and np.isfinite(y):
                px = (1 - x) if mirror else x
                cv2.circle(img, (int(px * w), int(y * h)), 3, color, -1)


def put_lines(img: np.ndarray, lines: list[str], origin=(10, 25)) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/tcn_last.pt"))
    parser.add_argument("--holistic-model", type=Path,
                        default=Path("models/holistic_landmarker.task"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None,
                        help="read frames from this file instead of the camera")
    parser.add_argument("--save", type=Path, default=None, help="write the annotated output (.mp4)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="frames per prediction")
    parser.add_argument("--stride", type=int, default=5, help="run the model every N frames")
    parser.add_argument("--smooth", type=int, default=5, help="predictions in the majority vote")
    parser.add_argument("--threshold", type=float, default=0.5, help="min mean probability")
    parser.add_argument("--mirror", action="store_true", help="flip the displayed image only")
    args = parser.parse_args()

    if not args.holistic_model.exists():
        raise SystemExit(f"MediaPipe model not found: {args.holistic_model}\n"
                         "Download holistic_landmarker.task (MediaPipe Holistic Landmarker) "
                         "and pass its path with --holistic-model.")

    model, epoch = load_model(args.ckpt)
    labels = load_labels()
    print(f"loaded {args.ckpt} (epoch {epoch}) | window {args.window} | stride {args.stride}")

    # VIDEO mode: synchronous, every frame in order, so the buffer holds consecutive frames
    landmarker = vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.holistic_model)),
        running_mode=vision.RunningMode.VIDEO,
    ))
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video or f'camera {args.camera}'}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps > 0:  # some files/cameras report 0; timestamps must still increase
        src_fps = 30.0
        print("warning: source reports no FPS, assuming 30")
    if args.video:
        print(f"video {args.video} | {src_fps:.2f} FPS | "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")

    buffer = deque(maxlen=args.window)
    smoother = Smoother(args.smooth, args.threshold)
    words = deque(maxlen=5)
    all_words = []  # full history for the end-of-video summary ('c' clears only the overlay)
    writer = None
    top3, latency_ms, fps = [], 0.0, 0.0
    n_frames, t_prev, t_start = 0, time.perf_counter(), time.monotonic()

    with torch.no_grad():
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # unflipped: dataset hands are not mirrored
            if args.video:  # file time, so results do not depend on processing speed
                timestamp_ms = int(n_frames * 1000 / src_fps)
            else:
                timestamp_ms = int((time.monotonic() - t_start) * 1000)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
            frame = result_to_frame(result)
            buffer.append(frame)
            n_frames += 1

            hands_in_buffer = any(has_hand(f) for f in buffer)
            if not hands_in_buffer:  # hands gone for a whole window: start a new word
                smoother.reset()
                top3 = []
            elif len(buffer) == args.window and n_frames % args.stride == 0:
                t0 = time.perf_counter()
                probs = model(window_to_input(buffer)).softmax(-1)[0]
                latency_ms = (time.perf_counter() - t0) * 1000
                p, c = probs.topk(3)
                top3 = list(zip(c.tolist(), p.tolist()))
                word = smoother.update(*top3[0])
                if word is not None:
                    words.append(labels[word])
                    all_words.append(labels[word])

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)  # smoothed camera FPS
            t_prev = now

            display = cv2.flip(bgr, 1) if args.mirror else bgr.copy()
            draw_landmarks(display, frame, args.mirror)
            put_lines(display, [f"FPS {fps:.1f} | model {latency_ms:.1f} ms"]
                      + [f"{labels[c]}: {p:.2f}" for c, p in top3]
                      + ["words: " + " ".join(words)])
            if args.save:
                if writer is None:
                    h, w = display.shape[:2]
                    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (w, h))
                writer.write(display)
            cv2.imshow("ASL realtime", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                words.clear()
                smoother.reset()

    cap.release()
    if writer is not None:
        writer.release()
        print(f"saved {args.save}")
    landmarker.close()
    cv2.destroyAllWindows()
    if args.video:
        print(f"processed {n_frames} frames | emitted words ({len(all_words)}): {all_words}")


if __name__ == "__main__":
    main()
