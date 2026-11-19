# TruthLens architecture

## Design principle
TruthLens separates **technical observations**, **model predictions**, and **human conclusions**. The MVP currently implements technical media inspection only. The detector adapter explicitly reports `not_configured`; it does not generate real/fake labels.

## Request flow
1. React sends one or more media files to `POST /api/analyze`.
2. FastAPI validates count, extension and byte size.
3. Each upload is written to a temporary file and removed after inspection.
4. The media inspector extracts image metadata, samples video frames, or reads supported audio signal statistics.
5. A detector adapter is called. Until a validated model is registered, it returns `not_configured`.
6. The API creates a case report with limitations and human-review requirements.
7. React displays observations and clearly labels the result inconclusive.

## Media support in the starter
- Image: JPEG, PNG, WEBP, BMP, TIFF.
- Video: MP4, MOV, AVI, MKV, WEBM (decoder support depends on the local OpenCV/FFmpeg build).
- Audio: WAV, FLAC, OGG (supported by the installed libsndfile build).

## Adding real models
Implement a modality-specific adapter behind `app/detectors/base.py`. Each adapter should return:
- model identifier and exact version;
- model-card and license reference;
- prediction output and raw scores;
- preprocessing details;
- known limitations and supported media domain.

Do not merge model scores into one probability unless calibration and fusion have been evaluated on a representative held-out dataset. Keep per-modality results separate and use a distinct `inconclusive` outcome.

## Production hardening
For a public deployment, move media decoding into isolated worker processes/containers with CPU, memory and time limits; use private object storage, retention policies, authentication, rate limiting, malware scanning, and audit logs. The local hackathon MVP is not a hardened public upload service.
