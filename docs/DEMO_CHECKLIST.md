# TruthLens demo checklist

## Before the demo
- Start FastAPI from the `backend` directory with `py -m uvicorn app.main:app --reload`.
- Start Vite from the `frontend` directory with `npm run dev`.
- Open `http://127.0.0.1:8000/health` and `http://localhost:5173`.
- Use short, non-sensitive sample files that you have permission to analyze.

## Suggested demo sequence
1. Upload a JPEG/PNG and show dimensions and available EXIF metadata.
2. Upload a short video and show duration, frame count and sampled timestamps.
3. Upload a short WAV/FLAC/OGG file and show sample rate, channels and signal statistics.
4. Upload a video and a separate audio file in one case to show the cross-modal human-review note.
5. Explain that no trained authenticity detector is connected in this starter; do not present metadata statistics as deepfake predictions.

## Expected status
The report should say `inconclusive_model_not_configured`. This is intentional and safer than fabricating a real/fake score.
