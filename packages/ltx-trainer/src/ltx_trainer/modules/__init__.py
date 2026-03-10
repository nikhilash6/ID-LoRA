"""
Custom modules for LTX-2 training.
"""

from ltx_trainer.modules.speaker_identity_adapter import (
    SpeakerIdentityAdapter,
    SpeakerIdentityAdapterConfig,
    SpeakerIdentityCrossAttention,
)

__all__ = [
    "SpeakerIdentityAdapter",
    "SpeakerIdentityAdapterConfig",
    "SpeakerIdentityCrossAttention",
]
