"""
Simulated Angiogram MJPEG Streaming Server.

Reads a video file (MP4, AVI, or rendered DICOM-export) and streams it as a
multipart MJPEG feed over HTTP, emulating the kind of live video feed that
would come from a capture card plugged into a cath lab monitor.

Usage:
    python src/scripts/simulate_stream.py --video path/to/angiogram.mp4
    python src/scripts/simulate_stream.py --video path/to/angiogram.mp4 --port 8080 --fps 15

The stream is served at:
    http://localhost:<port>/stream

Any standard MJPEG client can connect, including OpenCV:
    cap = cv2.VideoCapture("http://localhost:8080/stream")

Features:
    - Loops the video indefinitely so the stream never ends.
    - Respects the original video FPS, or overrides it with --fps.
    - Serves a simple status page at http://localhost:<port>/
"""

import argparse
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2


class StreamState:
    """Shared state between the frame-producer thread and the HTTP handler."""

    def __init__(self, video_path: str, fps: float):
        self.video_path = video_path
        self.fps = fps
        self.lock = threading.Lock()
        self.frame_jpeg: bytes = b""
        self.frame_count: int = 0
        self.running = True


def frame_producer(state: StreamState):
    """
    Background thread: reads frames from the video file at the configured FPS
    and encodes each one as a JPEG into shared state.  Loops the video
    indefinitely.
    """
    interval = 1.0 / state.fps
    while state.running:
        cap = cv2.VideoCapture(state.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {state.video_path}")
            state.running = False
            return

        while state.running:
            ret, frame = cap.read()
            if not ret:
                break  # end of file → loop

            # Encode the frame as JPEG
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with state.lock:
                state.frame_jpeg = jpeg.tobytes()
                state.frame_count += 1

            time.sleep(interval)

        cap.release()

    print("[INFO] Frame producer stopped.")


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves the MJPEG stream and a status page."""

    # Injected by the server setup
    stream_state: StreamState = None  # type: ignore

    def log_message(self, fmt, *args):
        # Suppress per-request logs to keep the terminal clean during streaming
        pass

    def do_GET(self):
        if self.path == "/stream":
            self._serve_stream()
        elif self.path == "/":
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_status(self):
        """Simple HTML status page so a browser can verify the server is running."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Angio-AI Stream Simulator</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0B0C0E; color: #F2F3F5;
         display: flex; flex-direction: column; align-items: center; padding: 40px; }}
  h1 {{ color: #BAED91; }}
  .card {{ background: #151719; border: 1px solid #2A2D32; border-radius: 12px;
           padding: 24px 32px; margin: 20px; max-width: 640px; width: 100%; }}
  code {{ background: #1D2024; padding: 4px 8px; border-radius: 4px; color: #BAED91; }}
  img {{ border-radius: 8px; margin-top: 16px; max-width: 100%; }}
  .label {{ color: #8A8F98; font-size: 13px; margin-bottom: 4px; }}
  .value {{ font-size: 15px; font-weight: 600; }}
</style>
</head>
<body>
  <h1>Angio-AI Stream Simulator</h1>
  <div class="card">
    <div class="label">Source Video</div>
    <div class="value">{self.stream_state.video_path}</div>
    <br>
    <div class="label">Target FPS</div>
    <div class="value">{self.stream_state.fps:.1f}</div>
    <br>
    <div class="label">Frames Served</div>
    <div class="value">{self.stream_state.frame_count}</div>
    <br>
    <div class="label">Stream URL</div>
    <div class="value"><code>http://localhost:{self.server.server_address[1]}/stream</code></div>
  </div>
  <div class="card">
    <div class="label">Live Preview</div>
    <img src="/stream" alt="Live stream preview">
  </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_stream(self):
        """
        Sends an infinite multipart MJPEG response.  Each part is one JPEG
        frame, delimited by a boundary string.  This is the standard format
        that OpenCV's VideoCapture and most MJPEG clients expect.
        """
        BOUNDARY = "angio_ai_frame"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        interval = 1.0 / self.stream_state.fps
        last_count = -1

        try:
            while self.stream_state.running:
                with self.stream_state.lock:
                    jpeg = self.stream_state.frame_jpeg
                    count = self.stream_state.frame_count

                if count == last_count or not jpeg:
                    time.sleep(0.005)
                    continue

                last_count = count
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                time.sleep(interval * 0.5)  # pace slightly to avoid busy-spin

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client disconnected — normal


def main():
    parser = argparse.ArgumentParser(
        description="Simulated MJPEG streaming server for Angio-AI live analysis demo."
    )
    parser.add_argument("--video", required=True, help="Path to the source video file (MP4, AVI, etc.)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port to serve the stream on (default: 8080)")
    parser.add_argument("--fps", type=float, default=0, help="Override playback FPS (0 = use video's native FPS)")
    args = parser.parse_args()

    # Determine FPS
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open video: {args.video}")
        return
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fps = args.fps if args.fps > 0 else native_fps

    print("=" * 60)
    print("  Angio-AI Stream Simulator")
    print("=" * 60)
    print(f"  Video   : {args.video}")
    print(f"  Size    : {width}x{height}  |  {total_frames} frames")
    print(f"  FPS     : {fps:.1f} (native: {native_fps:.1f})")
    print(f"  Port    : {args.port}")
    print(f"  Stream  : http://localhost:{args.port}/stream")
    print(f"  Status  : http://localhost:{args.port}/")
    print("=" * 60)
    print("  Press Ctrl+C to stop.\n")

    state = StreamState(args.video, fps)

    # Start the frame producer thread
    producer = threading.Thread(target=frame_producer, args=(state,), daemon=True)
    producer.start()

    # Configure the HTTP handler with our shared state
    MJPEGHandler.stream_state = state

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        state.running = False
        producer.join(timeout=3)
        server.server_close()


if __name__ == "__main__":
    main()
