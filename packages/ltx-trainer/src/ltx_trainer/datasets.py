from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


def pad_audio_latents_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function that pads variable-length audio latents.

    Audio latents may have slightly different sequence lengths (e.g., 120, 121, 122)
    due to variations in audio duration during preprocessing. This collate function
    pads them to the maximum length in the batch.

    Args:
        batch: List of sample dictionaries from PrecomputedDataset

    Returns:
        Collated batch with padded audio latents
    """
    # Keys that contain audio latents (need padding)
    audio_keys = {"audio_latents", "ref_audio_latents"}

    # First pass: find max audio sequence length for each audio key
    max_seq_lens: dict[str, int] = {}
    for key in audio_keys:
        if key in batch[0] and "latents" in batch[0][key]:
            max_len = max(sample[key]["latents"].shape[1] for sample in batch)  # T dimension
            max_seq_lens[key] = max_len

    # Second pass: pad audio latents and stack
    result: dict[str, Any] = {}

    for key in batch[0]:
        if key == "idx":
            # Stack indices directly
            result[key] = torch.tensor([sample[key] for sample in batch])
        elif key in audio_keys and key in max_seq_lens:
            # Handle audio latent dictionaries with padding
            max_len = max_seq_lens[key]
            inner_result: dict[str, Any] = {}

            # Get all inner keys from first sample
            inner_keys = batch[0][key].keys()
            for inner_key in inner_keys:
                values = [sample[key][inner_key] for sample in batch]

                if inner_key == "latents":
                    # Pad latents to max length: [C, T, F] -> pad T dimension
                    padded_values = []
                    for v in values:
                        current_len = v.shape[1]
                        if current_len < max_len:
                            # Pad on the right side of T dimension
                            # Shape: [C, T, F] - pad T (dim=1)
                            pad_size = max_len - current_len
                            v = F.pad(v, (0, 0, 0, pad_size), mode="constant", value=0)
                        padded_values.append(v)
                    inner_result[inner_key] = torch.stack(padded_values)
                elif inner_key == "num_time_steps":
                    # Store original (unpadded) lengths
                    if isinstance(values[0], int):
                        inner_result[inner_key] = torch.tensor(values)
                    else:
                        inner_result[inner_key] = torch.stack([torch.tensor(v) for v in values])
                elif isinstance(values[0], torch.Tensor):
                    inner_result[inner_key] = torch.stack(values)
                elif isinstance(values[0], (int, float)):
                    inner_result[inner_key] = torch.tensor(values)
                else:
                    # For strings and other non-stackable types, keep as list
                    inner_result[inner_key] = values

            result[key] = inner_result
        elif isinstance(batch[0][key], dict):
            # Handle other nested dictionaries (latents, conditions, etc.)
            inner_result = {}
            for inner_key in batch[0][key]:
                values = [sample[key][inner_key] for sample in batch]
                if isinstance(values[0], torch.Tensor):
                    inner_result[inner_key] = torch.stack(values)
                elif isinstance(values[0], (int, float)):
                    inner_result[inner_key] = torch.tensor(values)
                else:
                    # Keep as list for strings, etc.
                    inner_result[inner_key] = values
            result[key] = inner_result
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([sample[key] for sample in batch])
        else:
            # Keep other types as lists
            result[key] = [sample[key] for sample in batch]

    return result


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),  # random text embeddings
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),  # random attention mask
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources."""
        # Use first data source as the reference to discover samples
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]
        data_files = list(data_path.glob("**/*.pt"))

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        # Initialize sample files dict
        sample_files = {output_key: [] for output_key in self.data_sources.values()}

        # For each data file, find corresponding files in other sources
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check if corresponding files exist in ALL sources
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        """Check if corresponding files exist in all data sources."""
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                logger.warning(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False

        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        # Verify all output keys have the same number of samples
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                # Add file path info for strategies that need to access original media
                data["_file_path"] = str(file_path)
                data["_rel_path"] = str(file_rel_path)

                result[output_key] = data
            except Exception as e:
                logger.warning(f"Skipping corrupt sample {index} ({file_path}): {e}")
                import random

                fallback_index = random.randint(0, len(self) - 1)
                return self[fallback_index]

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data
