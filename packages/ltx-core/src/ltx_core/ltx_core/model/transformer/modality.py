from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    # Cross-modal attention masks for IC-LoRA training
    # a2v_cross_attention_mask: [B, video_seq, audio_seq] - masks which audio tokens video can attend to
    # v2a_cross_attention_mask: [B, audio_seq, video_seq] - masks which video tokens audio can attend to
    a2v_cross_attention_mask: torch.Tensor | None = None
    v2a_cross_attention_mask: torch.Tensor | None = None
    # Audio-to-text cross-attention mask for IC-LoRA training
    # audio_context_attn_mask: [B, audio_seq, text_seq] - masks which text tokens audio can attend to
    # Used to block reference audio tokens from attending to text conditioning
    audio_context_attn_mask: torch.Tensor | None = None