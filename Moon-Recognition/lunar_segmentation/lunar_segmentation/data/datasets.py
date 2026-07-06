import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
from pathlib import Path

class MoonTileDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, augment: bool = False, cache: bool = True):
        """
        cache: pre-load all tiles into RAM on init.
               Eliminates per-batch disk I/O + decompression for every epoch after the first.
               15k tiles × ~192 KB each ≈ 3 GB; disable if RAM is tight.
        """
        self.index_df = index_df.reset_index(drop=True)
        self.augment = augment
        self._cache: list = []

        if cache:
            print(f"Caching {len(self.index_df)} tiles into RAM…", flush=True)
            for _, row in self.index_df.iterrows():
                data = np.load(row['tile_path'])
                self._cache.append((
                    data['image'].astype(np.float32),
                    data['mask'].astype(np.float32),
                ))
            print("Cache ready.", flush=True)

    def __len__(self):
        return len(self.index_df)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask

    def __getitem__(self, idx):
        if self._cache:
            image, mask = self._cache[idx]
            image, mask = image.copy(), mask.copy()
        else:
            row = self.index_df.iloc[idx]
            data = np.load(row['tile_path'])
            image = data['image'].astype(np.float32)
            mask = data['mask'].astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)
        return torch.from_numpy(image), torch.from_numpy(mask)
