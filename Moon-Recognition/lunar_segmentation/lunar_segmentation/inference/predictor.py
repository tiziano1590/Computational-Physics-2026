import numpy as np
import torch
import logging
from pathlib import Path
from ..data.preprocessing import iter_tile_origins, CLASS_NAMES

logger = logging.getLogger(__name__)


def _remap_doubleconv_key(key: str) -> str:
    """Map legacy DoubleConv keys to the current architecture.

    Early checkpoints stored each DoubleConv as one ``nn.Sequential`` named
    ``block`` (``block.0/1`` = first Conv+BN, ``block.3/4`` = second Conv+BN).
    The current DoubleConv uses two named sub-Sequentials ``conv1``/``conv2``.
    Without this remap the published weights match *no* layer names and, under
    ``strict=False``, every conv silently keeps its random initialisation.
    """
    return (key
            .replace('.block.0', '.conv1.0')
            .replace('.block.1', '.conv1.1')
            .replace('.block.3', '.conv2.0')
            .replace('.block.4', '.conv2.1'))


class Predictor:
    def __init__(
        self,
        model,
        weights_path: Path = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)

        if weights_path:
            # weights_only=False: torch>=2.6 defaults to True, which rejects our
            # checkpoints (they embed a pandas metrics DataFrame next to the
            # state dict). Only load checkpoints produced by this project.
            weights = torch.load(str(weights_path), map_location=self.device,
                                 weights_only=False)
            if isinstance(weights, dict):
                # 'model' = legacy local checkpoints; 'state_dict' = the
                # CloudVeneto training-script checkpoints (with 'results' df)
                for key in ('state_dict', 'model'):
                    if key in weights:
                        weights = weights[key]
                        break
            flat = {k.replace('model.', '', 1): v for k, v in weights.items()}
            # Normalise legacy DoubleConv 'block.*' naming to the current
            # 'conv1'/'conv2' layout before loading (see _remap_doubleconv_key),
            # otherwise these checkpoints silently load nothing under strict=False.
            flat = {_remap_doubleconv_key(k): v for k, v in flat.items()}
            # strict=False tolerates legacy checkpoints (e.g. saved without the
            # residual projection layers), but a silent mismatch would mean
            # predicting with randomly initialised layers — so surface it loudly.
            result = self.model.load_state_dict(flat, strict=False)
            if result.missing_keys or result.unexpected_keys:
                logger.warning(
                    f"Checkpoint/architecture mismatch loading {weights_path}: "
                    f"missing={result.missing_keys} unexpected={result.unexpected_keys}. "
                    "Check use_residual / base_width match the checkpoint."
                )
            logger.info(f"Loaded weights from {weights_path}")

    def predict(
        self,
        image_chw: np.ndarray,
        tile_size: int = 256,
        stride: int = 128,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Sliding-window inference over a single large tile.

        Window defaults (256/128) match the training tile size and stride, so
        the model sees the same spatial context at inference as it did during
        training.  NOTE: the south-pole results in the report were generated
        with the previous defaults (128/64); pass those explicitly to reproduce
        them exactly.

        Instead of calling the model once per window (the original approach),
        windows are batched together and sent to the GPU/MPS in groups of
        `batch_size`.  For a 512×512 input with stride=64 this reduces ~49
        separate forward passes to a single batched call.

        Args:
            image_chw:  preprocessed 3-channel input (C, H, W).
            tile_size:  spatial size of each inference window.
            stride:     step between consecutive windows.
            batch_size: number of windows per GPU batch.

        Returns:
            Probability map (n_classes, H, W) averaged over all overlapping windows.
        """
        self.model.eval()
        n_classes = len(CLASS_NAMES)
        _, h, w = image_chw.shape

        prob_sum  = np.zeros((n_classes, h, w), dtype=np.float32)
        count_sum = np.zeros((h, w),            dtype=np.float32)

        # Collect valid window origins up-front
        origins = [
            (r, c)
            for r, c in iter_tile_origins(h, w, tile_size, stride)
            if image_chw[:, r:r+tile_size, c:c+tile_size].shape[1:] == (tile_size, tile_size)
        ]

        with torch.no_grad():
            for start in range(0, len(origins), batch_size):
                batch_origins = origins[start : start + batch_size]

                # Stack windows → (B, C, tile_size, tile_size)
                patches = np.stack(
                    [image_chw[:, r:r+tile_size, c:c+tile_size] for r, c in batch_origins]
                ).astype(np.float32)

                x     = torch.from_numpy(patches).to(self.device)
                probs = torch.sigmoid(self.model(x)).cpu().numpy()  # (B, n_classes, ts, ts)

                for (r, c), p in zip(batch_origins, probs):
                    prob_sum[:, r:r+tile_size, c:c+tile_size] += p
                    count_sum[r:r+tile_size, c:c+tile_size]   += 1.0

        return prob_sum / np.clip(count_sum[np.newaxis], 1.0, None)
