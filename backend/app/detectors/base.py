from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any


@dataclass
class DetectorResult:
    status: str
    model_name: str | None
    observations: list[dict[str, Any]]
    note: str


class Detector(Protocol):
    def analyze(self, path: Path) -> DetectorResult: ...


class UnconfiguredDetector:
    """Honest placeholder: never fabricates a real/fake prediction."""

    def analyze(self, path: Path) -> DetectorResult:
        return DetectorResult(
            status="not_configured",
            model_name=None,
            observations=[],
            note=(
                "No validated detector is connected yet. This report contains "
                "technical metadata observations only, not an authenticity verdict."
            ),
        )
