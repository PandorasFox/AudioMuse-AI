# tasks/sonic_backends/musicnn.py
"""Default sonic backend: MusiCNN embedding + MusiCNN-prediction tag head.

This is the historical AudioMuse path. It is the default — set
``SONIC_BACKEND=musicnn`` or leave the variable unset to keep behavior
unchanged. When ``musicnn`` is the active backend the orchestrator runs
``tasks.analysis.song.analyze_track`` directly (this backend's
``analyze`` is not on the hot path); the backend exists so the registry
always has a default entry and so ``load_sessions``/``cleanup_sessions``
route uniformly for every backend.

The actual ONNX inference (patch extraction, the two-stage
embedding+prediction models, the OOM→CPU fallback dance, and the
``sigmoid(mean(sigmoid(logits)))`` mood pipeline) lives in
``tasks.analysis.song`` and is delegated to here so there is a single
source of truth. The ``song`` module is imported lazily inside methods
to avoid an import cycle (``song`` imports this registry at analyze
time).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import SonicBackend, register

logger = logging.getLogger(__name__)


class MusiCNNBackend(SonicBackend):
    name = "musicnn"
    embedding_dim = 200
    target_sr = 16000

    def __init__(self) -> None:
        # Defer config import to instance time so unit tests that stub
        # ``config`` after import still pick up the active values.
        from config import EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH
        self._model_paths = {
            "embedding": EMBEDDING_MODEL_PATH,
            "prediction": PREDICTION_MODEL_PATH,
        }

    # --- session management ------------------------------------------------

    def load_sessions(self) -> Dict[str, Any]:
        from ..analysis import song
        sessions = song.load_musicnn_sessions(self._model_paths)
        if sessions is None:
            raise RuntimeError("Failed to load MusiCNN ONNX sessions")
        return sessions

    def cleanup_sessions(self, sessions: Dict[str, Any], context: str = "") -> None:
        try:
            from ..analysis import song
            song.cleanup_musicnn_sessions(sessions, context=context)
        except Exception as e:  # noqa: BLE001
            logger.warning("MusiCNN cleanup raised (suppressed): %s", e)

    # --- per-track analysis ------------------------------------------------

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        sessions: Optional[Dict[str, Any]],
        *,
        file_basename: str,
        mood_labels: List[str],
    ) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        embedding, moods = _run_musicnn_tags(
            audio, sr, sessions, self._model_paths,
            file_basename=file_basename, mood_labels=mood_labels,
        )
        if embedding is None:
            return None
        return embedding.astype(np.float32, copy=False), moods


def _run_musicnn_tags(
    audio: np.ndarray,
    sr: int,
    sessions: Optional[Dict[str, Any]],
    model_paths: Dict[str, str],
    *,
    file_basename: str,
    mood_labels: List[str],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """Run MusiCNN over ``audio`` and return ``(track_embedding, moods)``.

    Thin wrapper over ``tasks.analysis.song`` primitives so the MERT
    backend can reuse the MusiCNN tag head while substituting only the
    stored similarity embedding. Returns ``(None, None)`` when the track
    is too short to form a spectrogram patch or inference fails.
    """
    from ..analysis import song
    patches = song._patches_for_track(audio, sr, file_basename)
    if patches is None:
        return None, None
    return song._run_musicnn_models(
        patches, mood_labels, model_paths, sessions, file_basename
    )


register(MusiCNNBackend())
