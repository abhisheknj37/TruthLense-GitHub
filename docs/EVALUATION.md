# TruthLens evaluation protocol

## Do not claim accuracy without measurements
Before describing a detector as reliable, evaluate it on a labeled dataset with a subject/source-disjoint train/validation/test split. Avoid mixing near-duplicate frames from the same source across splits.

## Minimum evaluation
For each modality, record:
- dataset name, version, license and provenance;
- number of real and manipulated samples;
- manipulation types and capture conditions;
- precision, recall, F1 and ROC-AUC where appropriate;
- false-positive rate at the chosen operating threshold;
- confusion matrix and confidence calibration;
- performance on unseen sources, recompression, resizing and noise.

For video, report both clip-level and frame-level behavior, and explain how sampled frames are aggregated. For audio, evaluate across codecs, languages, speakers and recording environments. For cross-modal review, use paired data with known synchronization or transcript mismatch labels.

## Reporting rules
- A model score is not proof of authenticity.
- Avoid wording such as "100% real" or "definitely fake".
- Report "inconclusive" when model support is absent, inputs are outside the model's domain, or modalities disagree.
- Preserve original files and record hashes if chain-of-custody requirements apply.
- Surface the evidence and limitations so a human reviewer can reproduce the finding.

## Current MVP status
No authenticity model is wired in. Current outputs are metadata, signal measurements and video frame sampling only. These are useful forensic context but are not deepfake detection.
