from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.ml_runtime import heavy_models_enabled


def temporal_video_forensics(path: Path, max_samples: int = 120) -> dict[str, Any]:
    """Compute sequentially decoded temporal review cues; not a trained deepfake verdict."""
    from app.services.media import sample_video_frames
    try:
        decoded = sample_video_frames(path, min(max(2, max_samples), 24))
    except Exception:
        return {"status": "unavailable", "reason": "Video decode failed."}
    fps = decoded["fps"]
    samples = decoded["sampled_frames"]
    if fps is None or len(samples) < 3:
        return {"status": "insufficient_data", "reason": "At least three decoded frames and valid FPS are required."}
    previous_gray, previous_hist = None, None
    transitions: list[dict[str, float]] = []
    cuts: list[float] = []
    for sample in samples:
        frame = cv2.resize(sample["frame"], (320, 180))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        timestamp = sample["timestamp_seconds"]
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            distance = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            transitions.append({
                "timestamp_seconds": timestamp,
                "mean_flow": round(float(np.mean(magnitude)), 4),
                "flow_p95": round(float(np.percentile(magnitude, 95)), 4),
                "histogram_distance": round(distance, 4),
            })
            if distance >= .65:
                cuts.append(timestamp)
        previous_gray, previous_hist = gray, hist

    values = [row["mean_flow"] for row in transitions]
    discontinuities: list[float] = []
    if len(values) >= 4:
        median = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median))) + 1e-6
        discontinuities = [row["timestamp_seconds"] for row in transitions if row["mean_flow"] > median + 4.5 * mad]
    return {
        "status": "completed",
        "decoded_frame_count": decoded["frame_count"],
        "sampled_frames": len(samples),
        "method": "Two-pass exact temporal stratification; Farneback optical flow and HSV histogram change analysis",
        "sampled_transitions": len(transitions),
        "scene_change_timestamps_seconds": cuts,
        "motion_discontinuity_timestamps_seconds": discontinuities,
        "motion_summary": {
            "mean": round(float(np.mean(values)), 4) if values else None,
            "std": round(float(np.std(values)), 4) if values else None,
        },
        "samples": transitions,
        "trained_model_assessment": _run_temporal_torchscript(path),
        "interpretation": "Abrupt motion or scene changes are review cues only. Camera movement, edits, cuts, lighting and compression can create the same signals; this is not a trained temporal deepfake verdict.",
    }

def _run_temporal_torchscript(path: Path) -> dict[str, Any]:
    if not heavy_models_enabled():
        return {"status": "skipped_low_memory", "note": "Optional temporal ML inference is disabled on this low-memory host."}
    model_path = os.getenv("TRUTHLENS_TEMPORAL_MODEL", "").strip()
    if not model_path or not Path(model_path).is_file():
        return {"status": "model_not_configured", "required_configuration": "TRUTHLENS_TEMPORAL_MODEL", "note": "No validated temporal classifier configured; heuristic results remain review cues."}
    from app.services.media import sample_video_frames
    try:
        decoded = sample_video_frames(path, 16)
        samples = decoded["sampled_frames"]
        if len(samples) < 8:
            return {"status": "insufficient_frames", "decoded_frame_count": decoded["frame_count"]}
        frames = [
            cv2.cvtColor(cv2.resize(sample["frame"], (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            for sample in samples
        ]
    except Exception as exc:
        return {"status": "decode_error", "detail": type(exc).__name__}
    try:
        import torch
        tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)
        model = torch.jit.load(model_path, map_location="cpu").eval()
        with torch.inference_mode():
            output = model(tensor)
        if not isinstance(output, torch.Tensor) or output.numel() != 1:
            return {
                "status": "unsupported_model_output",
                "model_path": Path(model_path).name,
                "output_shape": list(output.shape) if isinstance(output, torch.Tensor) else None,
                "note": "The configured temporal adapter requires exactly one scalar output. No class score was selected from a multi-value output.",
            }
        score = float(output.reshape(-1)[0].item())
        if not np.isfinite(score):
            return {
                "status": "invalid_model_output",
                "model_path": Path(model_path).name,
                "note": "The configured temporal model returned a non-finite scalar; no score was reported.",
            }
        return {
            "status": "model_assessed",
            "model_path": Path(model_path).name,
            "raw_model_output": score,
            "score_semantics": "unverified_scalar_output",
            "calibration_status": "not_calibrated",
            "note": "A scalar output was returned, but its class mapping, activation and calibration are not inferred from its numeric range. It is not a probability or manipulation verdict.",
        }
    except Exception as exc:
        return {"status": "inference_error", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}

def face_identity_consistency(path: Path, max_samples: int = 60) -> dict[str, Any]:
    """Compare face embeddings only within a clip; this function never identifies people."""
    if not heavy_models_enabled():
        return {"status": "skipped_low_memory", "note": "Optional face-embedding inference is disabled on this low-memory host."}
    if importlib.util.find_spec("insightface") is None:
        return {"status": "dependency_missing", "required_package": "insightface", "note": "Optional InsightFace dependency is not installed; no face consistency comparison was performed."}
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=os.getenv("TRUTHLENS_FACE_MODEL", "buffalo_l"), providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as exc:
        return {"status": "model_unavailable", "note": f"Face embedding model unavailable: {type(exc).__name__}"}
    from app.services.media import sample_video_frames
    try:
        decoded = sample_video_frames(path, min(max(2, max_samples), 32))
    except Exception:
        return {"status": "unavailable", "note": "Video decode failed."}
    rows = []
    for sample in decoded["sampled_frames"]:
        faces = app.get(sample["frame"])
        if faces:
            face = max(faces, key=lambda item: float((item.bbox[2]-item.bbox[0]) * (item.bbox[3]-item.bbox[1])))
            rows.append((sample["timestamp_seconds"], np.asarray(face.normed_embedding, dtype=np.float32)))
    if len(rows) < 2:
        return {"status": "insufficient_faces", "faces_compared": len(rows), "decoded_frame_count": decoded["frame_count"], "note": "At least two sampled face detections are required."}
    similarities = [float(np.dot(rows[i-1][1], rows[i][1])) for i in range(1, len(rows))]
    try:
        threshold = float(os.getenv("TRUTHLENS_FACE_SIMILARITY_REVIEW_THRESHOLD", "0.35"))
        if not 0.0 <= threshold <= 1.0:
            threshold = 0.35
    except ValueError:
        threshold = 0.35
    points = [{"timestamp_seconds": rows[i][0], "cosine_similarity_to_previous_sample": round(similarities[i-1], 4)} for i in range(1, len(rows)) if similarities[i-1] < threshold]
    return {"status": "completed", "method": "within-video normalized face-embedding similarity", "faces_compared": len(rows), "decoded_frame_count": decoded["frame_count"], "adjacent_similarity_mean": round(float(np.mean(similarities)), 4), "review_threshold": threshold, "identity_change_review_points": points, "note": "This does not identify anyone or prove a face swap. Pose, occlusion, lighting, multiple people and detector errors can change embeddings; threshold is heuristic."}

def syncnet_lipsync(path_video: Path, path_audio: Path | None = None) -> dict[str, Any]:
    """Run optional SyncNet only when package, checkpoint and FFmpeg are configured."""
    if not heavy_models_enabled():
        return {"status": "skipped_low_memory", "note": "Optional SyncNet inference is disabled on this low-memory host."}
    weights = os.getenv("TRUTHLENS_SYNCNET_WEIGHTS", "").strip()
    if not weights or not Path(weights).is_file():
        return {"status": "model_weights_missing", "required_configuration": "TRUTHLENS_SYNCNET_WEIGHTS", "note": "No lip-sync score was produced because compatible SyncNet weights are not configured."}
    if importlib.util.find_spec("syncnet_python") is None:
        return {"status": "dependency_missing", "required_package": "syncnet-python", "note": "Install the optional SyncNet integration before use."}
    try:
        from syncnet_python import SyncNetInstance
        class Args:
            batch_size = 20
            vshift = 10
            reference = path_video.stem[:64]
        with tempfile.TemporaryDirectory(prefix="truthlens_syncnet_") as tmp:
            Args.tmp_dir = tmp
            analysis_video = path_video
            if path_audio is not None:
                muxed = Path(tmp) / "paired_evidence.mp4"
                command = ["ffmpeg", "-y", "-i", str(path_video), "-i", str(path_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(muxed)]
                mux = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if mux.returncode != 0 or not muxed.exists():
                    return {"status": "mux_failed", "note": "FFmpeg could not prepare the selected audiovisual pair; original files were not changed.", "detail": mux.stderr[-400:]}
                analysis_video = muxed
            model = SyncNetInstance()
            model.loadParameters(weights)
            offset, confidence, distances = model.evaluate(Args, str(analysis_video))
        return {"status": "model_assessed", "model": "SyncNet audio-visual synchronization", "estimated_offset_frames": int(offset), "sync_confidence": float(confidence), "minimum_distance": float(min(distances)) if len(distances) else None, "paired_audio_used": path_audio is not None, "note": "Model-specific synchronization measure, not probability of manipulation. Dubbing, edits, offsets and multiple speakers can also affect results."}
    except Exception as exc:
        return {"status": "inference_error", "model": "SyncNet", "detail": f"{type(exc).__name__}: {str(exc)[:200]}", "note": "No valid lip-sync result was produced."}


def verify_c2pa(path: Path) -> dict[str, Any]:
    """Read C2PA credentials with the optional official SDK; report validation metadata conservatively."""
    if not path.is_file():
        return {
            "status": "evidence_file_missing",
            "standard": "C2PA Content Credentials",
            "note": "The evidence file is unavailable at the supplied path, so C2PA inspection was not attempted. No provenance conclusion was made.",
        }
    if importlib.util.find_spec("c2pa") is None:
        return {"status": "validator_not_installed", "standard": "C2PA Content Credentials", "note": "Install c2pa-python to enable manifest validation."}
    mime = {".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png", ".webp":"image/webp", ".mp4":"video/mp4", ".mov":"video/quicktime"}.get(path.suffix.lower())
    if not mime:
        return {"status": "unsupported_format", "standard": "C2PA Content Credentials", "note": "This file extension is not supported by the configured C2PA reader."}
    try:
        from c2pa import Reader
    except Exception as exc:
        # The optional native C2PA library can be blocked by Windows
        # Application Control / DLL policy even when the Python package exists.
        return {
            "status": "validator_unavailable",
            "standard": "C2PA Content Credentials",
            "detail": f"{type(exc).__name__}: {str(exc)[:250]}",
            "note": "The C2PA Python package is present, but its native validator could not be loaded. Check Windows Application Control policy or use a supported runtime. No provenance conclusion was made.",
        }
    try:
        with path.open("rb") as stream:
            with Reader(mime, stream) as reader:
                raw = reader.json()
        manifest = json.loads(raw) if isinstance(raw, str) else raw
        active = manifest.get("active_manifest")
        selected = (manifest.get("manifests") or {}).get(active, {}) if active else {}
        validation = manifest.get("validation_status", []) or selected.get("validation_status", [])
        return {"status": "manifest_found_validation_review", "standard": "C2PA Content Credentials", "active_manifest": active, "claim_generator": selected.get("claim_generator"), "validation_status": validation, "note": "Review SDK validation results and configured trust anchors. Manifest presence alone does not establish trusted provenance or truthfulness."}
    except Exception as exc:
        error_name = type(exc).__name__.lower()
        error_text = str(exc).lower()
        manifest_absent = (
            "manifestnotfound" in error_name
            or "manifest not found" in error_text
            or "no jumbf data found" in error_text
        )
        if manifest_absent:
            return {
                "status": "manifest_not_present",
                "standard": "C2PA Content Credentials",
                "detail": "No embedded C2PA/JUMBF manifest was found.",
                "note": "No C2PA Content Credentials were detected. This does not indicate manipulation or establish authenticity; creator and editing history cannot be verified through C2PA for this file.",
            }
        return {
            "status": "verification_error",
            "standard": "C2PA Content Credentials",
            "detail": f"{type(exc).__name__}: {str(exc)[:250]}",
            "note": "The C2PA verification process encountered an error. No trusted provenance conclusion can be drawn.",
        }
