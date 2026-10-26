"""Camera for the patrol, with a webcam fallback for PC development.

Deliberately does NOT reuse Code/Server/camera.py. That class is built around
picamera2's JpegEncoder and exposes frames only as JPEG bytes through a
`get_frame()` that calls `condition.wait()` with no timeout and always waits
for a *new* frame -- so it blocks forever if streaming was never started
(camera.py:75-79). Neither the encoder nor that API suits an inference loop.

Instead the patrol configures picamera2 itself with no encoder at all and
pulls BGR arrays straight from the ISP:

  * 640x480 `main` stream in XBGR8888, one `capture_array` per inference.
  * 640x480 letterboxes into the model's 640x640 with padding ONLY and zero
    downscaling. The model was trained at imgsz=640, so this is the cheapest
    accuracy in the whole project. Upstream's 400x300 stream (main.py:46)
    would force a downscale and cost small-object recall -- which is exactly
    what a helmet is at patrol distance.
  * No JpegEncoder. The live view needs *annotated* frames anyway, so JPEG
    encoding happens in ppe_mjpeg from the already-drawn BGR frame, and only
    while someone is actually watching. The Pi 5 has no hardware video
    encoder, so avoiding a second software encode is worth real CPU.

Because the patrol owns the camera outright, there is no contention with
upstream's video thread. Picamera2 is exclusive: run stock `main.py` for
manual driving OR `ppe_main.py` for patrol, never both at once.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class CameraSource(Protocol):
    def start(self) -> None: ...
    def capture_bgr(self) -> np.ndarray | None: ...
    def close(self) -> None: ...
    @property
    def size(self) -> tuple[int, int]: ...


class PiCamera:
    """picamera2, no encoder, BGR arrays straight from the ISP."""

    def __init__(
        self,
        size: tuple[int, int] = (640, 480),
        hflip: bool = False,
        vflip: bool = False,
        buffer_count: int = 4,
        max_fps: int = 30,
    ) -> None:
        self._size = (int(size[0]), int(size[1]))
        self.hflip = hflip
        self.vflip = vflip
        self.buffer_count = buffer_count
        self.max_fps = max_fps
        self.camera = None
        self._started = False

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def start(self) -> None:
        if self._started:
            return
        from libcamera import Transform
        from picamera2 import Picamera2

        self.camera = Picamera2()
        # XBGR8888 gives a 4-channel array we drop the alpha from -- one
        # cvtColor, no colour-space guessing. A `lores` stream would be forced
        # to YUV420 and need a heavier conversion for no benefit here.
        config = self.camera.create_video_configuration(
            main={"size": self._size, "format": "XBGR8888"},
            transform=Transform(
                hflip=1 if self.hflip else 0, vflip=1 if self.vflip else 0
            ),
            buffer_count=self.buffer_count,
            # Cap the sensor rate: we only consume a few FPS, and an idle ISP
            # running at 60 is wasted heat and power on a battery robot.
            controls={"FrameDurationLimits": (int(1e6 / self.max_fps), 200000)},
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)  # let AE/AWB settle, or the first frames are dark
        self._started = True
        print(f"[camera] picamera2 started at {self._size[0]}x{self._size[1]}")

    def capture_bgr(self) -> np.ndarray | None:
        if not self._started:
            return None
        try:
            import cv2

            arr = self.camera.capture_array("main")
            if arr is None:
                return None
            if arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            return arr
        except Exception as exc:
            print(f"[camera] capture failed: {exc}")
            return None

    def close(self) -> None:
        if self.camera is not None:
            try:
                if self._started:
                    self.camera.stop()
                self.camera.close()
            except Exception as exc:
                print(f"[camera] close failed: {exc}")
        self._started = False
        self.camera = None


class WebcamCamera:
    """cv2.VideoCapture, so the vision stack can be developed on the PC.

    Not a toy: it lets the detector, rule engine, tracker and confirmation
    logic all be exercised against a real moving scene with a real person,
    hours before the robot is involved. Only the drive layer needs hardware.
    """

    def __init__(self, size: tuple[int, int] = (640, 480), index: int = 0) -> None:
        self._size = (int(size[0]), int(size[1]))
        self.index = index
        self.cap = None

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def start(self) -> None:
        import cv2

        # CAP_DSHOW avoids a multi-second open on Windows' default backend.
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam {self.index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        actual = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != self._size:
            print(f"[camera] webcam gave {actual}, asked for {self._size}")
            self._size = actual
        print(f"[camera] webcam {self.index} started at {self._size}")

    def capture_bgr(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None


class FileCamera:
    """Replay a video file or loop a single image. For deterministic testing."""

    def __init__(self, path: str, size: tuple[int, int] = (640, 480), loop: bool = True):
        self.path = path
        self._size = size
        self.loop = loop
        self.cap = None
        self.still = None

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def start(self) -> None:
        import cv2

        still = cv2.imread(self.path)
        if still is not None:
            self.still = still
            self._size = (still.shape[1], still.shape[0])
            print(f"[camera] looping still image {self.path} at {self._size}")
            return
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.path} as an image or video")
        self._size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        print(f"[camera] replaying {self.path} at {self._size}")

    def capture_bgr(self) -> np.ndarray | None:
        import cv2

        if self.still is not None:
            return self.still.copy()
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None


def open_camera(cfg, source: str = "auto", index: int = 0, path: str | None = None):
    """Build a camera from config.

    `source`: "auto" prefers picamera2 and falls back to a webcam, which makes
    the same command work on the Pi and on a dev PC.
    """
    size = tuple(cfg.camera.stream_size)

    if source == "file":
        if not path:
            raise ValueError("source=file requires a path")
        return FileCamera(path, size)
    if source == "webcam":
        return WebcamCamera(size, index)
    if source == "picamera":
        return PiCamera(size, cfg.camera.hflip, cfg.camera.vflip, cfg.camera.buffer_count)

    try:
        import picamera2  # noqa: F401

        return PiCamera(size, cfg.camera.hflip, cfg.camera.vflip, cfg.camera.buffer_count)
    except ImportError:
        print("[camera] picamera2 unavailable; falling back to a webcam")
        return WebcamCamera(size, index)


if __name__ == "__main__":
    import argparse

    import cv2

    from ppe_config import load_config

    ap = argparse.ArgumentParser(description="Grab frames and check the colour path.")
    ap.add_argument("--source", default="auto",
                    choices=("auto", "picamera", "webcam", "file"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--path")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--out", default="camera_test.jpg")
    args = ap.parse_args()

    cfg = load_config()
    cam = open_camera(cfg, args.source, args.index, args.path)
    cam.start()
    try:
        frame = None
        t0 = time.perf_counter()
        got = 0
        for _ in range(args.frames):
            f = cam.capture_bgr()
            if f is not None:
                frame = f
                got += 1
        elapsed = time.perf_counter() - t0

        if frame is None:
            raise SystemExit("no frames captured")

        h, w = frame.shape[:2]
        print(f"captured {got}/{args.frames} frames, {w}x{h}, "
              f"{got / max(elapsed, 1e-6):.1f} fps")
        b, g, r = (float(frame[:, :, i].mean()) for i in range(3))
        print(f"channel means  B={b:.1f} G={g:.1f} R={r:.1f}")
        print(
            "  A strongly blue-looking scene with a high R mean (or vice versa) "
            "means the BGR/RGB path is inverted."
        )
        cv2.imwrite(args.out, frame)
        print(f"wrote {args.out} -- open it and check the colours look right")
    finally:
        cam.close()
