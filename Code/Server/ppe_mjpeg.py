"""Minimal MJPEG server, for the portal's live view.

Serves the latest *annotated* frame -- boxes and PPE verdicts already drawn by
the vision thread -- so what you watch in the portal is literally what the
detector sees. That makes it a debugging tool rather than decoration: a
threshold that is too low or an association that is going wrong is obvious on
screen within seconds.

Deliberately not the Freenove video protocol on port 8000. That is a raw
socket carrying `struct.pack('<I', len)` followed by JPEG bytes
(main.py:308-323) and only its own PyQt client can read it. MJPEG over HTTP is
viewable in any browser and on a phone, and the portal proxies it so there is
a single address to remember.

Costs nothing when nobody is watching: frames are JPEG-encoded lazily, only
while at least one client is connected.

Bound to 0.0.0.0 on purpose. Upstream's `Server.get_interface_ip()` resolves
wlan0 and silently falls back to 127.0.0.1 (server.py:15-22), which binds
loopback-only and is invisible from the network -- a failure that looks like a
dead robot.
"""

from __future__ import annotations

import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

BOUNDARY = "ppeframe"


class FrameBuffer:
    """Latest annotated frame, shared between the vision thread and clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._condition = threading.Condition(self._lock)
        self.viewers = 0
        self.status_line = ""

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._seq += 1
            self._condition.notify_all()

    def wait_for_frame(self, last_seq: int, timeout: float = 5.0):
        """Block until a frame newer than `last_seq`. Returns (jpeg, seq)."""
        with self._condition:
            if self._seq == last_seq:
                self._condition.wait(timeout)
            return self._jpeg, self._seq

    @property
    def has_viewers(self) -> bool:
        return self.viewers > 0

    @property
    def seq(self) -> int:
        return self._seq


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    buffer: FrameBuffer = None  # type: ignore[assignment]

    def log_message(self, *_args) -> None:
        pass  # one line per frame would be unreadable

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_page()
        elif path in ("/stream.mjpg", "/stream"):
            self._send_stream()
        elif path == "/snapshot.jpg":
            self._send_snapshot()
        elif path == "/healthz":
            self._send_bytes(b'{"ok":true}', "application/json")
        else:
            self.send_error(404)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self) -> None:
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PPE patrol live</title>
<style>
 body{{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center;gap:10px;padding:16px}}
 img{{max-width:100%;border-radius:8px}}
 code{{color:#8ab4ff}}
</style></head>
<body>
 <h2>PPE patrol &mdash; live</h2>
 <img src="/stream.mjpg" alt="live view">
 <p>{self.buffer.status_line or "waiting for frames"}</p>
 <p><code>/stream.mjpg</code> &middot; <code>/snapshot.jpg</code></p>
</body></html>"""
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_snapshot(self) -> None:
        jpeg, _ = self.buffer.wait_for_frame(-1, timeout=3.0)
        if jpeg is None:
            self.send_error(503, "no frame yet")
            return
        self._send_bytes(jpeg, "image/jpeg")

    def _send_stream(self) -> None:
        self.buffer.viewers += 1
        try:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.end_headers()

            last_seq = -1
            while True:
                jpeg, seq = self.buffer.wait_for_frame(last_seq, timeout=5.0)
                if jpeg is None:
                    time.sleep(0.2)
                    continue
                if seq == last_seq:
                    continue  # timed out waiting; loop so we notice disconnects
                last_seq = seq
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # viewer closed the tab; entirely normal
        except Exception as exc:
            print(f"[mjpeg] stream ended: {type(exc).__name__}: {exc}")
        finally:
            self.buffer.viewers = max(0, self.buffer.viewers - 1)


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class MjpegServer:
    """Runs the HTTP server on its own thread."""

    def __init__(self, buffer: FrameBuffer, port: int = 8080, host: str = "0.0.0.0"):
        self.buffer = buffer
        self.port = port
        self.host = host
        self._server: _ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = type("Handler", (_Handler,), {"buffer": self.buffer})
        try:
            self._server = _ThreadedHTTPServer((self.host, self.port), handler)
        except OSError as exc:
            print(f"[mjpeg] cannot bind {self.host}:{self.port}: {exc}")
            print("[mjpeg] live view disabled; patrol continues")
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="ppe-mjpeg", daemon=True
        )
        self._thread.start()
        print(f"[mjpeg] live view on http://{self.host}:{self.port}/stream.mjpg")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None


def encode_jpeg(frame_bgr, quality: int = 70) -> bytes | None:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


if __name__ == "__main__":
    import argparse

    import cv2
    import numpy as np

    ap = argparse.ArgumentParser(
        description="Serve a test pattern over MJPEG, to check the portal proxy."
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()

    buffer = FrameBuffer()
    buffer.status_line = "test pattern"
    server = MjpegServer(buffer, port=args.port)
    server.start()
    print("Ctrl-C to stop. Point the portal at this with:")
    print(f"  set PPE_ROBOT_MJPEG=http://127.0.0.1:{args.port}/stream.mjpg")

    try:
        frame_no = 0
        while True:
            frame = np.full((480, 640, 3), 40, dtype=np.uint8)
            x = int(240 + 180 * np.sin(frame_no / 12.0))
            cv2.rectangle(frame, (x, 150), (x + 150, 420), (96, 104, 120), -1)
            cv2.circle(frame, (x + 75, 180), 36, (188, 152, 126), -1)
            cv2.rectangle(frame, (x, 150), (x + 150, 420), (60, 60, 240), 2)
            cv2.putText(frame, "NO HELMET (test pattern)", (x, 142),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 240), 2)
            cv2.putText(frame, f"frame {frame_no}  viewers {buffer.viewers}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            jpeg = encode_jpeg(frame)
            if jpeg:
                buffer.publish(jpeg)
            frame_no += 1
            time.sleep(1.0 / max(args.fps, 0.5))
    except KeyboardInterrupt:
        server.stop()
