# VisualSearch

A local-first industrial video research service. The browser app accepts any video, samples it into an indexed frame set, sends an investigator's natural-language question plus those frames to a vision-language model, and navigates the player to returned evidence timestamps.

This repository is deliberately split into two planes:

- `server.py` is the local ingestion, indexing, and VLM retrieval service.
- `index.html` is the reviewer experience and local file playback.

## Run it

`ffmpeg` and `ffprobe` must be available on `PATH`. Configure any OpenAI-compatible multimodal chat endpoint, then start the service:

```powershell
$env:VISION_API_URL = "https://your-provider.example/v1/chat/completions"
$env:VISION_MODEL = "your-vision-model"
$env:VISION_API_KEY = "optional-provider-key"
python server.py
```

Open `http://127.0.0.1:4174`, upload a video, wait for indexing, and ask a question. Videos and sampled frames remain in `data/` locally.

## Vercel deployment

Deploy `index.html`, `config.js`, and `vercel.json` to Vercel as a static web app. Set `window.VISUALSEARCH_API_BASE` in `config.js` to the public URL of this Python worker before deployment. Vercel is intentionally not used for upload or inference: its function-body and execution limits make it unsuitable for long video processing. Keep the worker on a GPU-capable host with `ffmpeg`, this service, and the free YOLO runtime.

## Production path

1. Split uploaded video into sampled frames and retain source-time offsets.
2. Run detector + tracker (YOLO/RT-DETR + ByteTrack/BoT-SORT) and persist object tracks.
3. Add zone geometry and rule/temporal-event extraction for crossings, PPE, carrying, arrivals, and falls.
4. Index frame/window embeddings (SigLIP/CLIP or a video-language model) alongside structured events.
5. Turn the user query into structured constraints plus an embedding, then fuse event filtering with temporal retrieval.
6. Return evidence-window URLs, bounding boxes, scores, model/version provenance, and timestamps to this interface.

For long or high-volume footage, replace the frame sampler with a queue-backed worker fleet and add detector/tracker outputs plus a persistent vector/event store. The current service is a working single-machine research baseline, not a substitute for that distributed pipeline.
