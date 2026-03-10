"""Training strategies for different conditioning modes.

This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio)
- Audio-ref-only IC training (ID-LoRA: audio-only reference with negative positions)

Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

from ltx_trainer import logger
from ltx_trainer.training_strategies.audio_ref_only_ic import AudioRefOnlyICConfig, AudioRefOnlyICStrategy
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    LossResult,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy

TrainingStrategyConfig = (
    TextToVideoConfig
    | AudioRefOnlyICConfig
)

__all__ = [
    "AudioRefOnlyICConfig",
    "AudioRefOnlyICStrategy",
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "LossResult",
    "ModelInputs",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy."""

    match config:
        case TextToVideoConfig():
            strategy = TextToVideoStrategy(config)
        case AudioRefOnlyICConfig():
            strategy = AudioRefOnlyICStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    if isinstance(config, AudioRefOnlyICConfig):
        audio_mode = "(audio enabled)"
    else:
        audio_mode = "(audio enabled)" if getattr(config, "with_audio", False) else "(audio disabled)"
    logger.debug(f"Using {strategy.__class__.__name__} training strategy {audio_mode}")
    return strategy
