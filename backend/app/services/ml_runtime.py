"""Runtime guard for heavyweight local ML inference on small hosting plans."""
import os


def heavy_models_enabled() -> bool:
    """Heavy PyTorch/Transformers inference is opt-in on constrained hosts."""
    value = os.getenv("TRUTHLENS_ENABLE_HEAVY_MODELS", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def skipped_model_result(modality: str, model_name: str) -> dict:
    return {
        "status": "model_skipped_low_memory",
        "model_name": model_name,
        "prediction": "inconclusive",
        "uncertainty_band": "unknown",
        "scores": {},
        "note": (
            f"The {modality} ML detector was not run because "
            "TRUTHLENS_ENABLE_HEAVY_MODELS is disabled for this low-memory host. "
            "Lightweight forensic checks may still be available. No real/fake "
            "classification was made."
        ),
    }
