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


def temporal_video_forensics(path: Path, max_samples: int = 120) -> dict[str, Any]:
    """Compute temporal review cues; these are not a trained deepfake verdict."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "unavailable", "reason": "Video decode failed."}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total < 3:
        cap.release()
        return {"status": "insufficient_data", "reason": "At least three frames and valid FPS are required."}
    step = max(1, total // max(1, max_samples))
    previous_gray, previous_hist = None, None
    transitions: list[dict[str, float]] = []
    cuts: list[float] = []
    try:
        for index in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            timestamp = round(index / fps, 3)
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
    finally:
        cap.release()

    values = [row["mean_flow"] for row in transitions]
    discontinuities: list[float] = []
    if len(values) >= 4:
        median = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median))) + 1e-6
        discontinuities = [row["timestamp_seconds"] for row in transitions if row["mean_flow"] > median + 4.5 * mad]
    return {
        "status": "completed",
        "method": "Farneback optical flow and HSV histogram change analysis",
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
    model_path = os.getenv("TRUTHLENS_TEMPORAL_MODEL", "").strip()
    if not model_path or not Path(model_path).is_file():
        return {"status": "model_not_configured", "required_configuration": "TRUTHLENS_TEMPORAL_MODEL", "note": "No validated temporal classifier configured; heuristic results remain review cues."}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "decode_error"}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    try:
        if total < 8:
            return {"status": "insufficient_frames"}
        for index in np.linspace(0, total - 1, 16, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
    finally:
        cap.release()
    if len(frames) < 8:
        return {"status": "insufficient_frames"}
    try:
        import torch
        tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)
        model = torch.jit.load(model_path, map_location="cpu").eval()
        with torch.inference_mode():
            output = model(tensor)
        score = float(output.reshape(-1)[0].item())
        return {"status": "model_assessed", "model_path": Path(model_path).name, "raw_fake_score": score, "score_kind_assumption": "probability" if 0 <= score <= 1 else "logit", "note": "Only use weights whose architecture, preprocessing, label mapping and evaluation provenance match the tensor contract. Uncalibrated output is not a forensic probability."}
    except Exception as exc:
        return {"status": "inference_error", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}


def face_identity_consistency(path: Path, max_samples: int = 60) -> dict[str, Any]:
    """Compare face embeddings only within a clip; this function never identifies people."""
    if importlib.util.find_spec("insightface") is None:
        return {"status": "dependency_missing", "required_package": "insightface", "note": "Optional InsightFace dependency is not installed; no face consistency comparison was performed."}
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=os.getenv("TRUTHLENS_FACE_MODEL", "buffalo_l"), providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as exc:
        return {"status": "model_unavailable", "note": f"Face embedding model unavailable: {type(exc).__name__}"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "unavailable", "note": "Video decode failed."}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    rows = []
    try:
        if total <= 0:
            return {"status": "insufficient_data", "note": "No frames available."}
        for index in np.linspace(0, total - 1, min(max_samples, total), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = app.get(frame)
            if faces:
                face = max(faces, key=lambda item: float((item.bbox[2]-item.bbox[0]) * (item.bbox[3]-item.bbox[1])))
                rows.append((round(index / fps, 3) if fps else None, np.asarray(face.normed_embedding, dtype=np.float32)))
    finally:
        cap.release()
    if len(rows) < 2:
        return {"status": "insufficient_faces", "faces_compared": len(rows), "note": "At least two sampled face detections are required."}
    similarities = [float(np.dot(rows[i-1][1], rows[i][1])) for i in range(1, len(rows))]
    try:
        threshold = float(os.getenv("TRUTHLENS_FACE_SIMILARITY_REVIEW_THRESHOLD", "0.35"))
        if not 0.0 <= threshold <= 1.0:
            threshold = 0.35
    except ValueError:
        threshold = 0.35
    points = [{"timestamp_seconds": rows[i][0], "cosine_similarity_to_previous_sample": round(similarities[i-1], 4)} for i in range(1, len(rows)) if similarities[i-1] < threshold]
    return {"status": "completed", "method": "within-video normalized face-embedding similarity", "faces_compared": len(rows), "adjacent_similarity_mean": round(float(np.mean(similarities)), 4), "review_threshold": threshold, "identity_change_review_points": points, "note": "This does not identify anyone or prove a face swap. Pose, occlusion, lighting, multiple people and detector errors can change embeddings; threshold is heuristic."}


def syncnet_lipsync(path_video: Path, path_audio: Path | None = None) -> dict[str, Any]:
    """Run optional SyncNet only when package, checkpoint and FFmpeg are configured."""
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
        return {"status": "verification_error", "standard": "C2PA Content Credentials", "detail": f"{type(exc).__name__}: {str(exc)[:250]}", "note": "No trusted provenance conclusion can be drawn."}
