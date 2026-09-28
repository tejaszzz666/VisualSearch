"""Local VisualSearch service: upload, frame indexing, and VLM-backed evidence retrieval."""

from __future__ import annotations

import base64
import cgi
import json
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

ROOT = Path(__file__).parent
STORE = ROOT / "data"
VIDEOS = STORE / "videos"
FRAMES = STORE / "frames"
JOBS: dict[str, dict] = {}
DETECTOR = None


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()


def video_duration(path: Path) -> float:
    try:
        return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0.0


def enhance_frame(path: Path) -> dict:
    """Improve low-contrast, undersized footage before either detector sees it."""
    image = Image.open(path).convert("RGB")
    edges = ImageStat.Stat(image.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]
    enhanced = ImageOps.autocontrast(image, cutoff=1)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.25)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6 if edges < 1200 else 1.15)
    if min(enhanced.size) < 720:
        scale = min(2, 720 / min(enhanced.size))
        enhanced = enhanced.resize((round(enhanced.width * scale), round(enhanced.height * scale)), Image.Resampling.LANCZOS)
    enhanced.save(path, quality=92)
    return {"clarity": round(min(100, edges / 35), 1), "enhanced": edges < 1200}


def detect_frame(path: Path) -> list[dict]:
    """Run a free YOLO detector at a high enough resolution for small PPE/objects."""
    global DETECTOR
    if YOLO is None:
        return []
    if DETECTOR is None:
        DETECTOR = YOLO(os.environ.get("YOLO_MODEL", "yolo11m.pt"))
    result = DETECTOR.predict(source=str(path), imgsz=1280, conf=0.18, verbose=False)[0]
    names = result.names
    return [
        {"label": names[int(box.cls[0])], "confidence": round(float(box.conf[0]) * 100), "box": [round(float(value), 1) for value in box.xyxy[0].tolist()]}
        for box in result.boxes
    ]


def index_video(job_id: str, video_path: Path) -> None:
    job = JOBS[job_id]
    try:
        job.update(status="indexing", progress="Sampling evidence frames")
        duration = video_duration(video_path)
        frame_dir = FRAMES / job_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        interval = max(5, int(duration / 30)) if duration else 10
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vf", f"fps=1/{interval},scale=960:-2", "-q:v", "4", str(frame_dir / "frame-%04d.jpg"),
        ], check=True)
        samples = []
        for i, frame in enumerate(sorted(frame_dir.glob("*.jpg"))):
            quality = enhance_frame(frame)
            samples.append({"seconds": round(i * interval, 1), "path": str(frame), "quality": quality, "detections": detect_frame(frame)})
        job.update(status="ready", progress="Ready for research", duration=duration, samples=samples)
    except FileNotFoundError:
        job.update(status="error", progress="ffmpeg and ffprobe are required to index video")
    except subprocess.CalledProcessError:
        job.update(status="error", progress="Could not decode this video file")
    except Exception as error:  # Keep the service useful even when a codec/model response surprises it.
        job.update(status="error", progress=str(error))


def decode_json_response(raw: str) -> list[dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
        return [item for item in parsed if isinstance(item, dict) and "seconds" in item]
    except json.JSONDecodeError:
        return []


def ask_vision_model(question: str, samples: list[dict]) -> list[dict]:
    endpoint = os.environ.get("VISION_API_URL")
    model = os.environ.get("VISION_MODEL")
    api_key = os.environ.get("VISION_API_KEY")
    if not endpoint or not model:
        raise RuntimeError("Set VISION_API_URL and VISION_MODEL to research indexed video.")

    evidence: list[dict] = []
    instruction = (
        "You are an industrial video investigator. Inspect the timestamped sampled frames and answer the question. "
        "Return only a JSON array. Each item must have seconds (number), label (short evidence description), "
        "confidence (0-100 integer), track (subject identity or description), object, and zone. "
        "Only return likely matches; never invent visibility that is not in the frames."
    )
    for batch_start in range(0, len(samples), 8):
        content: list[dict] = [{"type": "text", "text": f"{instruction}\nQuestion: {question}"}]
        for sample in samples[batch_start : batch_start + 8]:
            encoded = base64.b64encode(Path(sample["path"]).read_bytes()).decode("ascii")
            detection_hint = ", ".join(item["label"] for item in sample["detections"][:12]) or "no detector labels"
            content.append({"type": "text", "text": f"Timestamp: {sample['seconds']} seconds. YOLO objects: {detection_hint}. Enhancement applied: {sample['quality']['enhanced']}."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"}})
        payload = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}]}).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        with urlopen(Request(endpoint, data=payload, headers=headers), timeout=120) as response:
            response_json = json.loads(response.read())
        text = response_json["choices"][0]["message"]["content"]
        evidence.extend(decode_json_response(text))
    return sorted(evidence, key=lambda item: item["seconds"])


def summarize_video(samples: list[dict]) -> str:
    endpoint, model = os.environ.get("VISION_API_URL"), os.environ.get("VISION_MODEL")
    if not endpoint or not model:
        raise RuntimeError("Set VISION_API_URL and VISION_MODEL to create a video summary.")
    labels = []
    for sample in samples:
        labels.extend(item["label"] for item in sample["detections"])
    prompt = "Create a concise, factual video summary. Mention activity, subjects, notable objects, safety risks, and uncertainty. Do not claim anything not supported by the evidence. Detector observations: " + ", ".join(labels[:200])
    payload = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VISION_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['VISION_API_KEY']}"
    with urlopen(Request(endpoint, data=payload, headers=headers), timeout=120) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


class Handler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/index":
            self.start_index()
        elif self.path == "/api/research":
            self.research()
        elif self.path == "/api/summary":
            self.summary()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def start_index(self) -> None:
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")})
        uploaded = form["video"] if "video" in form else None
        if not uploaded or not getattr(uploaded, "file", None):
            self.reply(HTTPStatus.BAD_REQUEST, {"error": "A video file is required."})
            return
        job_id = uuid.uuid4().hex
        suffix = Path(uploaded.filename or "video.mp4").suffix or ".mp4"
        path = VIDEOS / f"{job_id}{suffix}"
        VIDEOS.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uploaded.file.read())
        JOBS[job_id] = {"id": job_id, "name": uploaded.filename, "status": "queued", "progress": "Queued for indexing", "samples": []}
        threading.Thread(target=index_video, args=(job_id, path), daemon=True).start()
        self.reply(HTTPStatus.ACCEPTED, JOBS[job_id])

    def research(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        job = JOBS.get(body.get("assetId"))
        if not job:
            self.reply(HTTPStatus.NOT_FOUND, {"error": "Video index not found."})
            return
        if job["status"] != "ready":
            self.reply(HTTPStatus.CONFLICT, {"error": job["progress"]})
            return
        try:
            matches = ask_vision_model(str(body.get("question", "")), job["samples"])
            self.reply(HTTPStatus.OK, {"matches": matches, "duration": job["duration"]})
        except Exception as error:
            self.reply(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    def summary(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        job = JOBS.get(body.get("assetId"))
        if not job or job["status"] != "ready":
            self.reply(HTTPStatus.CONFLICT, {"error": "Video must finish indexing before it can be summarized."})
            return
        try:
            self.reply(HTTPStatus.OK, {"summary": summarize_video(job["samples"])})
        except Exception as error:
            self.reply(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    def do_GET(self) -> None:
        if self.path.startswith("/api/jobs/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1])
            self.reply(HTTPStatus.OK if job else HTTPStatus.NOT_FOUND, job or {"error": "Job not found."})
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def reply(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    STORE.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", "4174"))
    print(f"VisualSearch running on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
