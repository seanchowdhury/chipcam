import io
import os
import time
import threading
import subprocess
from datetime import datetime

import numpy as np
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2.outputs import CircularOutput, FileOutput

# ── Config ────────────────────────────────────────────────────────────────────
MAIN_SIZE        = (1920, 1080)
LORES_SIZE       = (640, 360)
MOTION_THRESHOLD = 4          # mean luma diff (0–255) that triggers recording
MOTION_COOLDOWN  = 5          # seconds of no motion before clip is finalized
PRE_BUFFER_SECS  = 5          # seconds of footage buffered before motion
H264_BITRATE     = 4_000_000
CLIPS_DIR        = "clips"
PORT             = 5000

os.makedirs(CLIPS_DIR, exist_ok=True)

# ── MJPEG streaming buffer ────────────────────────────────────────────────────
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)

# ── Shared state ──────────────────────────────────────────────────────────────
app           = Flask(__name__)
stream_output = StreamingOutput()
circular: CircularOutput | None = None

_recording      = False
_manual_recording = False   # True when started via button, skips motion auto-stop
_last_motion_ts = 0.0
_prev_luma      = None
_motion_event   = threading.Event()
_state_lock     = threading.Lock()
_active_clip    = None

# ── Motion detection (runs in camera capture thread via pre_callback) ─────────
def motion_callback(request):
    global _prev_luma

    with MappedArray(request, "lores") as m:
        # YUV420 shape is (H*3//2, W); first H rows are the Y (luma) plane
        luma = m.array[:LORES_SIZE[1], :].astype(np.float32)

    if _prev_luma is not None:
        diff = np.abs(luma - _prev_luma).mean()
        if diff > MOTION_THRESHOLD:
            _motion_event.set()

    _prev_luma = luma

# ── Recording manager (dedicated thread) ──────────────────────────────────────
def recording_manager():
    global _recording, _last_motion_ts, _active_clip

    while True:
        triggered = _motion_event.wait(timeout=0.5)

        if triggered:
            _motion_event.clear()
            _last_motion_ts = time.time()

            with _state_lock:
                if not _recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _active_clip = os.path.join(CLIPS_DIR, f"clip_{timestamp}.h264")
                    circular.fileoutput = _active_clip
                    circular.start()
                    _recording = True
                    print(f"[motion] Recording → {_active_clip}")

        else:
            with _state_lock:
                if _recording and not _manual_recording and (time.time() - _last_motion_ts > MOTION_COOLDOWN):
                    circular.stop()
                    _recording = False
                    clip = _active_clip
                    print(f"[motion] Clip complete — converting {clip}")
                    threading.Thread(target=_convert_clip, args=(clip,), daemon=True).start()


def _convert_clip(h264_path: str):
    mp4_path = h264_path.replace(".h264", ".mp4")
    result = subprocess.run(
        ["ffmpeg", "-framerate", "30", "-i", h264_path, "-c", "copy", mp4_path, "-y"],
        capture_output=True,
    )
    if result.returncode == 0:
        os.remove(h264_path)
        print(f"[ffmpeg] Saved {mp4_path}")
    else:
        print(f"[ffmpeg] Error: {result.stderr.decode()}")

# ── Flask routes ──────────────────────────────────────────────────────────────
def _generate_frames():
    while True:
        with stream_output.condition:
            stream_output.condition.wait()
            frame = stream_output.frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    clips = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4"))
    return jsonify(recording=_recording, clips=clips)


@app.route("/record/start", methods=["POST"])
def record_start():
    global _recording, _manual_recording, _active_clip
    with _state_lock:
        if _recording:
            return jsonify(ok=False, reason="already recording")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _active_clip = os.path.join(CLIPS_DIR, f"clip_{timestamp}.h264")
        circular.fileoutput = _active_clip
        circular.start()
        _recording = True
        _manual_recording = True
        print(f"[manual] Recording → {_active_clip}")
    return jsonify(ok=True)


@app.route("/record/stop", methods=["POST"])
def record_stop():
    global _recording, _manual_recording, _active_clip
    with _state_lock:
        if not _recording:
            return jsonify(ok=False, reason="not recording")
        circular.stop()
        _recording = False
        _manual_recording = False
        clip = _active_clip
        print(f"[manual] Clip complete — converting {clip}")
        threading.Thread(target=_convert_clip, args=(clip,), daemon=True).start()
    return jsonify(ok=True)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    return send_from_directory(CLIPS_DIR, filename)


@app.route("/clips/<path:filename>/rename", methods=["POST"])
def rename_clip(filename):
    new_name = request.json.get("name", "").strip()
    if not new_name or "/" in new_name or "\\" in new_name:
        return jsonify(ok=False, reason="invalid name"), 400
    if not new_name.endswith(".mp4"):
        new_name += ".mp4"
    src = os.path.join(CLIPS_DIR, filename)
    dst = os.path.join(CLIPS_DIR, new_name)
    if not os.path.isfile(src):
        return jsonify(ok=False, reason="not found"), 404
    if os.path.exists(dst):
        return jsonify(ok=False, reason="name already taken"), 409
    os.rename(src, dst)
    return jsonify(ok=True, name=new_name)


@app.route("/clips/<path:filename>/delete", methods=["DELETE"])
def delete_clip(filename):
    path = os.path.join(CLIPS_DIR, filename)
    if not os.path.isfile(path):
        return jsonify(ok=False, reason="not found"), 404
    os.remove(path)
    return jsonify(ok=True)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global circular

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": MAIN_SIZE},
        lores={"size": LORES_SIZE, "format": "YUV420"},
    )
    picam2.configure(config)
    picam2.pre_callback = motion_callback

    h264_encoder  = H264Encoder(bitrate=H264_BITRATE)
    circular      = CircularOutput(buffersize=PRE_BUFFER_SECS * H264_BITRATE // 8)
    mjpeg_encoder = MJPEGEncoder()

    # H264 on main stream → circular buffer (always running, pre-motion footage)
    picam2.start_recording(h264_encoder, circular)
    # MJPEG on lores stream → web stream
    picam2.start_encoder(mjpeg_encoder, FileOutput(stream_output), name="lores")

    threading.Thread(target=recording_manager, daemon=True).start()

    print(f"[server] http://0.0.0.0:{PORT}")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        picam2.stop_recording()


if __name__ == "__main__":
    main()
