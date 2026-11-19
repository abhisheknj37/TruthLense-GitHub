# TruthLens architecture

## Evidence-first design
TruthLens separates technical observations, model outputs and human conclusions. Each uploaded object receives a server-computed SHA-256 digest and optional client/server transfer-integrity comparison. Processing uses a temporary working copy. Originals are removed after processing by default; an investigator can explicitly opt into private backend retention with expiry and authenticated digest-checked downloads.

## Modality pipelines

### Image
- Face locations are proposed by OpenCV Haar Cascade.
- Up to three largest face crops are passed to \`sakshamkr1/deitfake-v2\` (Apache-2.0 model card).
- The implementation reads the checkpoint's \`id2label\` mapping instead of assuming the class order.
- Output contains per-face real/fake softmax scores, a median summary, a heuristic inconclusive band and an optional transformer patch-attribution overlay.

**Scope:** the checkpoint is described as a face-cropped real/fake image classifier trained from OpenForensics-derived data. It is a research/educational model, not a validated forensic service. Images with no detected face remain inconclusive.

### Video
- OpenCV reads the container metadata.
- Up to sixteen deterministic, evenly spaced frame indices are selected after a first sequential pass counts actual decodable frames; a second pass extracts those exact indices.
- Each frame is passed through the same face-crop image detector.
- The report stores frame index, timestamp, per-frame result and a median score over face-bearing frames.

**Scope:** this is sampled frame classification, not temporal deepfake detection. It does not establish lip-sync, continuity or clip authenticity.

### Audio
- Audio is resampled to 16 kHz mono and analyzed in up to five 8-second windows.
- \`Hemgg/Deepfake-audio-detection\` is loaded with Transformers (Apache-2.0 model card).
- Class indices are resolved from the model config labels (\`AIVoice\` and \`HumanVoice\`), not guessed.
- Per-window scores and median audio score are returned.

**Scope:** the model card describes a multi-ethnic English audio dataset and a self-reported evaluation score, while also noting that more data are needed. This is a narrow research signal, not general voice-clone verification. It may fail on languages, codecs, generators or recording conditions outside its training data.

## Uncertainty and decision bands
The 0.35 / 0.65 cutoffs are configurable heuristic score bands. They are not calibrated confidence thresholds. A separate review-uncertainty record uses score separation, sample range/disagreement and sample count; those labels are heuristic, not probability estimates. A model/pipeline-specific calibration artifact can only be applied after source-group held-out validation and explicit reviewer approval. No labelled validation dataset or approved calibration artifact is bundled.

## Grad-CAM-style patch attribution
For the DeiT image model, the last transformer encoder block is hooked. The gradient of the fake-class logit with respect to patch tokens produces a patch-importance overlay. This visualization identifies image regions that influenced this model's output; it is not a manipulation mask and does not explain causality.

## Cross-modal comparison
Cross-modal comparison runs only after the user explicitly marks separate audio and video uploads as belonging to the same recording. It compares the audio-window signal with the video face-frame signal and reports agreement, disagreement or inconclusive status. It does not calculate a fused authenticity probability and does not independently verify synchronization or provenance. A separate relative-timeline model-score profile comparison is descriptive association, not a sync offset or authenticity result. Text-bearing evidence sources also receive pairwise lexical comparison; optional bidirectional NLI returns sentence relationship candidates, not verified claims.

## Run-time behavior
Models are loaded lazily on first use and cached in process memory. The first call may take time and download model weights from Hugging Face. CPU inference is supported but may be slow; CUDA is used when available.

## API
- \`GET /health\`
- \`GET /api/capabilities\`
- \`POST /api/analyze\` multipart fields: \`files\` (one or more), \`context_text\` (optional), \`pair_media\` (optional boolean).

## Public deployment hardening
The local hackathon MVP needs isolated media workers, process-level CPU/memory/time limits, stronger content validation, authentication, rate limiting, private storage and deletion policies before public exposure. Do not allow uploaded media to execute.


Plain-text evidence accepts UTF-8 TXT/MD files up to 1 MB. Optional NLI runs bidirectional sentence-pair checks when configured. Paired audio/video review also compares normalized relative-time model-score profiles; this is not a synchronization offset, fused score or authenticity conclusion.

Original evidence retention is opt-in per case. Retained files have an expiry manifest, per-user private storage and an authenticated download that rechecks SHA-256. The owner can request an expiry purge. The local custody ledger can optionally HMAC-sign event hashes when a server-only secret is configured.


## Text and source evidence
UTF-8 TXT/MD files (1 MB max) are imported as source evidence. OCR text from images/sampled video frames, optional ASR transcripts, supplied case context and text files are preserved as separate sources. Pairwise lexical overlap reports shared vocabulary without inferring truth or contradiction. If `TRUTHLENS_NLI_MODEL` is configured, local NLI compares bounded sentence pairs in both directions and reports uncalibrated entailment/contradiction/neutral candidates.

## Original retention and custody
Original retention is opt-in per case. Private copies use an owner-derived storage namespace, report/asset IDs, a sidecar expiry manifest and SHA-256 verification at download. The local hash-linked evidence ledger can optionally HMAC-sign event hashes with a server-only secret. This is tamper-evident local custody metadata, not an independent timestamp or external anchor.
