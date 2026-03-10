"""
Speaker Identity Adapter for Audio Branch

This module implements explicit speaker identity conditioning via cross-attention
in the audio branch of the LTX-2 transformer. It uses a pretrained WavLM+ECAPA-TDNN
encoder to extract speaker identity embeddings, which are then projected and injected
via dedicated cross-attention layers.

Architecture:
- WavLM+ECAPA-TDNN encoder (frozen) extracts 256-dim speaker embedding
- Projection layer maps embedding to transformer hidden dimension
- Per-block cross-attention: audio tokens attend to speaker identity token
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class SpeakerIdentityAdapterConfig:
    """Configuration for SpeakerIdentityAdapter."""
    
    enabled: bool = True
    hidden_dim: int = 3840  # Audio context dimension in LTX-2 (matches audio_prompt_embeds from connector)
    speaker_emb_dim: int = 256  # WavLM+ECAPA output dimension
    num_identity_tokens: int = 1  # Number of tokens to project speaker embedding into
    num_transformer_blocks: int = 32  # Number of transformer blocks in LTX-2
    num_heads: int = 16  # Number of attention heads
    wavlm_checkpoint_path: str | None = None  # Path to WavLM+ECAPA checkpoint


class SpeakerIdentityCrossAttention(nn.Module):
    """Cross-attention layer for speaker identity injection.
    
    Audio hidden states (Q) attend to speaker identity tokens (K, V).
    This allows each audio position to query "what should this sound like
    given the speaker identity?"
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Q from audio hidden states
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # K, V from speaker identity
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Output projection
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Layer norm for Q and K (following LTX-2 pattern)
        self.q_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        
        # Learnable scale (start at 0 for stable training)
        self.scale = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        audio_hidden: torch.Tensor,  # [B, seq_len, hidden_dim]
        speaker_tokens: torch.Tensor,  # [B, num_tokens, hidden_dim]
    ) -> torch.Tensor:
        """
        Apply cross-attention from audio to speaker identity.
        
        Returns:
            Updated audio hidden states with speaker identity information.
        """
        B, N, D = audio_hidden.shape
        
        # Compute Q, K, V
        q = self.to_q(audio_hidden)  # [B, N, D]
        k = self.to_k(speaker_tokens)  # [B, num_tokens, D]
        v = self.to_v(speaker_tokens)  # [B, num_tokens, D]
        
        # Apply layer norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Reshape for multi-head attention: [B, num_heads, seq_len, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        
        # Reshape back: [B, N, hidden_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        out = self.to_out(out)
        
        # Residual connection with learnable scale (tanh to bound it)
        return audio_hidden + self.scale.tanh() * out


class SpeakerIdentityAdapter(nn.Module):
    """Speaker Identity Adapter for audio branch.
    
    This adapter:
    1. Uses a frozen WavLM+ECAPA-TDNN encoder to extract speaker embeddings
    2. Projects embeddings to identity tokens that are prepended to audio text conditioning
    
    The identity tokens are prepended to audio_prompt_embeds, allowing the audio cross-attention
    (audio_attn2) to attend to both text and speaker identity. This is simpler than adding
    separate cross-attention layers and doesn't require modifying the transformer architecture.
    
    Usage during training:
    1. Extract speaker embedding from reference audio
    2. Project to identity tokens: [B, num_tokens, audio_context_dim]
    3. Prepend tokens to audio_prompt_embeds in the training strategy
    """
    
    def __init__(self, config: SpeakerIdentityAdapterConfig):
        super().__init__()
        self.config = config
        
        # Speaker encoder (loaded lazily, frozen)
        self._speaker_encoder = None
        self._encoder_device = None
        
        # Projection: speaker embedding -> identity tokens
        # Output shape: [B, num_tokens, hidden_dim] where hidden_dim = audio_context_dim (3840)
        self.speaker_proj = nn.Sequential(
            nn.Linear(config.speaker_emb_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim * config.num_identity_tokens),
        )
        
        # Normalize output to match text embedding scale (mean≈0, std≈1)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to produce outputs on similar scale to text embeddings.
        
        Text embeddings have mean≈0, std≈1.0, so we want identity tokens to 
        have similar scale to actually influence the attention.
        """
        for module in self.speaker_proj.modules():
            if isinstance(module, nn.Linear):
                # Use standard xavier initialization (gain=1.0) 
                # The small gain=0.1 was making identity tokens 10000x smaller than text
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _load_speaker_encoder(self, device: torch.device):
        """Lazy load the WavLM+ECAPA-TDNN encoder."""
        if self._speaker_encoder is not None and self._encoder_device == device:
            return
        
        from ltx_trainer.metrics.unispeech_sv import load_wavlm_ecapa_model
        
        checkpoint_path = self.config.wavlm_checkpoint_path
        if checkpoint_path is None:
            # Try default paths
            possible_paths = [
                Path("models/unispeech/wavlm_large_finetune.pth"),
                Path("/scratch/aviad/github/LTX-2/models/unispeech/wavlm_large_finetune.pth"),
            ]
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
        
        logger.info(f"Loading WavLM+ECAPA-TDNN speaker encoder from {checkpoint_path}")
        self._speaker_encoder = load_wavlm_ecapa_model(
            checkpoint_path=checkpoint_path,
            device=str(device),
        )
        self._encoder_device = device
        
        # Ensure encoder is frozen
        for param in self._speaker_encoder.parameters():
            param.requires_grad = False
        self._speaker_encoder.eval()
        
        logger.info("Speaker encoder loaded and frozen")
    
    def extract_speaker_embedding(
        self,
        audio_waveform: torch.Tensor,  # [B, num_samples] at 16kHz
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Extract speaker embedding from audio waveform.
        
        Args:
            audio_waveform: Audio tensor at 16kHz, shape [B, num_samples]
            device: Device to run encoder on (defaults to waveform device)
            
        Returns:
            Speaker embedding tensor of shape [B, 256]
        """
        if device is None:
            device = audio_waveform.device
        
        self._load_speaker_encoder(device)
        
        with torch.no_grad():
            # Ensure waveform is on correct device
            waveform = audio_waveform.to(device)
            
            # Extract embedding
            embedding = self._speaker_encoder(waveform)  # [B, 256]
            
            # Normalize
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def extract_speaker_embedding_from_file(
        self,
        audio_path: str | Path,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            device: Device for computation
            
        Returns:
            Speaker embedding tensor of shape [1, 256]
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Add batch dimension if needed: [1, num_samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] != 1:
            waveform = waveform.squeeze(0).unsqueeze(0)
        
        return self.extract_speaker_embedding(waveform, device)
    
    def project_to_identity_tokens(
        self,
        speaker_embedding: torch.Tensor,  # [B, 256]
    ) -> torch.Tensor:
        """Project speaker embedding to identity tokens for cross-attention.
        
        Args:
            speaker_embedding: Speaker embedding from WavLM, shape [B, 256]
            
        Returns:
            Identity tokens of shape [B, num_tokens, hidden_dim]
        """
        B = speaker_embedding.shape[0]
        
        # Project: [B, 256] -> [B, hidden_dim * num_tokens]
        projected = self.speaker_proj(speaker_embedding)
        
        # Reshape to tokens: [B, num_tokens, hidden_dim]
        identity_tokens = projected.view(B, self.config.num_identity_tokens, self.config.hidden_dim)
        
        # Normalize to match text embedding scale (mean≈0, std≈1)
        identity_tokens = self.output_norm(identity_tokens)
        
        return identity_tokens
    
    def offload_encoder_to_cpu(self):
        """Offload speaker encoder to CPU to free GPU memory.
        
        Call this after extracting embeddings for a batch to reduce memory usage
        during training.
        """
        if self._speaker_encoder is not None:
            logger.info("Offloading speaker encoder to CPU to free GPU memory")
            self._speaker_encoder = self._speaker_encoder.to("cpu")
            self._encoder_device = torch.device("cpu")
            torch.cuda.empty_cache()
    
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of trainable parameters (excludes frozen encoder)."""
        return list(self.speaker_proj.parameters()) + list(self.output_norm.parameters())
    
    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict with only trainable parameters (for saving checkpoints)."""
        state = {}
        # Save speaker_proj parameters
        for name, param in self.speaker_proj.named_parameters():
            state[f"speaker_proj.{name}"] = param
        # Save output_norm parameters
        for name, param in self.output_norm.named_parameters():
            state[f"output_norm.{name}"] = param
        return state
    
    def num_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())
