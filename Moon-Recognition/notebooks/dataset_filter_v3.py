"""
Filtraggio dataset lunare — v3.

Estende analizza_tile_v2 (notebook dataset_visualization_and_filtering.ipynb)
con tre famiglie di check aggiuntivi per catalogare *perché* una tile è
inutilizzabile:

  A) Mask:  mask_all_white, mask_block, mask_stripe
  B) Image: image_lowcontrast, image_dark, image_stripe
  C) Coerenza: misaligned (immagine OK ma maschera non corrisponde — recuperabile)

Lanciare dalla cartella notebooks/:
    python dataset_filter_v3.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

DATA_DIR = "../lunar_segmentation/data/MR"
INDEX_PATH = os.path.join(DATA_DIR, "tiles", "index.csv")
REPORT_PATH = os.path.join(DATA_DIR, "tiles", "corruption_report_v3.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "tiles", "index_clean_v3.csv")
RECOVER_PATH = os.path.join(DATA_DIR, "tiles", "index_recoverable_v3.csv")

# Soglie — tutte calibrabili a vista dopo aver visto le distribuzioni.
SOGLIE = {
    "mask_block_frac": 0.40,        # comp. connessa più grande / area tot.
    "mask_stripe_white_rows": 5,    # righe completamente bianche
    "mask_stripe_transitions": 2,   # transizioni bianco↔non-bianco
    "image_lowcontrast_std": 0.03,  # std globale dell'immagine
    "image_dark_thr": 0.02,         # soglia "pixel quasi nero"
    "image_dark_frac": 0.50,        # frazione di pixel quasi neri
    "image_stripe_fft": 8.0,        # peak / mean dell'energia per riga FFT
    "misaligned_score": 1.15,       # edge_in / edge_out
    "misaligned_min_craters": 3,    # range di n_craters in cui il test ha senso
    "misaligned_max_craters": 500,
}

CATEGORIE_ORDINATE = [
    "mask_all_white",
    "mask_block",
    "mask_stripe",
    "image_lowcontrast",
    "image_dark",
    "image_stripe",
    "misaligned",
    "normale",
    "errore",
]

COLOR_MAP = {
    "mask_all_white": "#e74c3c",
    "mask_block": "#c0392b",
    "mask_stripe": "#f39c12",
    "image_lowcontrast": "#9b59b6",
    "image_dark": "#34495e",
    "image_stripe": "#e67e22",
    "misaligned": "#3498db",
    "normale": "#2ecc71",
    "errore": "#95a5a6",
}


# ---------------------------------------------------------------------------
# Analisi singola tile
# ---------------------------------------------------------------------------

@dataclass
class TileReport:
    tipo: str = "normale"
    white_percent: float = 0.0
    n_white_rows: int = 0
    n_transitions: int = 0
    max_block_frac: float = 0.0
    n_craters: int = 0
    img_std: float = 0.0
    dark_frac: float = 0.0
    stripe_fft_score: float = 0.0
    align_score: float = float("nan")


def _mask_block_frac(mask: np.ndarray) -> float:
    """Frazione coperta dalla componente connessa di valori 1 più grande."""
    labeled, n = label(mask)
    if n == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    return float(sizes.max()) / mask.size


def _stripe_fft_score(img: np.ndarray) -> float:
    """Quanto l'energia FFT è concentrata su una singola freq. verticale."""
    centered = img - img.mean()
    F = np.fft.rfft2(centered)
    energy_v = np.abs(F).sum(axis=1)
    # Esclude DC e le frequenze più basse (gradienti lenti, non stripe)
    energy_v[:2] = 0
    if len(energy_v) > 2:
        energy_v[-2:] = 0
    mean = energy_v.mean()
    if mean <= 0:
        return 0.0
    return float(energy_v.max() / mean)


def _align_score(sobel: np.ndarray, mask: np.ndarray) -> float:
    """Edge medio dentro mask=1 / edge medio dentro mask=0."""
    inside = sobel[mask == 1]
    outside = sobel[mask == 0]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    edge_in = float(inside.mean())
    edge_out = float(outside.mean())
    if edge_out <= 1e-6:
        return float("nan")
    return edge_in / edge_out


def analizza_tile_v3(tile_path: str) -> TileReport:
    r = TileReport()
    try:
        data = np.load(os.path.join(DATA_DIR, tile_path))
        image = data["image"]      # (3, 256, 256)
        mask = data["mask"][0]     # canale crateri
        img0 = image[0]            # canale normalizzato
        sobel = image[2]           # canale Sobel pre-calcolato

        r.white_percent = float(np.mean(mask) * 100)

        # 1. Maschera tutta bianca
        if np.all(mask == 1):
            r.tipo = "mask_all_white"
            r.max_block_frac = 1.0
            return r

        # 2. Blocco bianco grande
        block_frac = _mask_block_frac(mask)
        r.max_block_frac = block_frac
        if block_frac > SOGLIE["mask_block_frac"]:
            r.tipo = "mask_block"
            # comunque calcoliamo gli altri segnali utili
            r.n_craters = int(label(mask)[1])
            return r

        # 3. Stripe sulla maschera
        row_means = mask.mean(axis=1)
        is_white = (row_means == 1.0)
        r.n_white_rows = int(is_white.sum())
        r.n_transitions = int(np.abs(np.diff(is_white.astype(int))).sum())
        if (r.n_white_rows > SOGLIE["mask_stripe_white_rows"]
                and r.n_transitions >= SOGLIE["mask_stripe_transitions"]):
            r.tipo = "mask_stripe"
            r.n_craters = int(label(mask)[1])
            return r

        # Pre-calcola stat immagine
        r.img_std = float(img0.std())
        r.dark_frac = float((img0 < SOGLIE["image_dark_thr"]).mean())
        r.n_craters = int(label(mask)[1])

        # 4. Immagine quasi uniforme
        if r.img_std < SOGLIE["image_lowcontrast_std"]:
            r.tipo = "image_lowcontrast"
            return r

        # 5. Immagine quasi tutta scura
        if r.dark_frac > SOGLIE["image_dark_frac"]:
            r.tipo = "image_dark"
            return r

        # 6. Banding orizzontale nell'immagine
        r.stripe_fft_score = _stripe_fft_score(img0)
        if r.stripe_fft_score > SOGLIE["image_stripe_fft"]:
            r.tipo = "image_stripe"
            return r

        # 7. Mismatch immagine ↔ maschera (solo se la maschera è "sensata")
        if (SOGLIE["misaligned_min_craters"] <= r.n_craters
                <= SOGLIE["misaligned_max_craters"]):
            r.align_score = _align_score(sobel, mask)
            if (not np.isnan(r.align_score)
                    and r.align_score < SOGLIE["misaligned_score"]):
                r.tipo = "misaligned"
                return r

        r.tipo = "normale"
        return r

    except Exception:
        r.tipo = "errore"
        return r


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analizza_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tile_path in tqdm(df["tile_path"], desc="Analisi tile"):
        rows.append(asdict(analizza_tile_v3(tile_path)))
    extra = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


def stampa_distribuzione(df: pd.DataFrame, etichetta: str = "v3"):
    print("\n" + "=" * 60)
    print(f"  DISTRIBUZIONE CATEGORIE — {etichetta}")
    print("=" * 60)
    conteggi = df["tipo"].value_counts()
    tot = len(df)
    for cat in CATEGORIE_ORDINATE:
        if cat in conteggi:
            n = int(conteggi[cat])
            print(f"  {cat:20s}: {n:6d}  ({n / tot * 100:5.1f}%)")
    print("-" * 60)
    print(f"  {'TOTALE':20s}: {tot:6d}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------

def _open_tile(tile_path: str):
    data = np.load(os.path.join(DATA_DIR, tile_path))
    return data["image"][0], data["mask"][0]


def visualizza_categoria(df: pd.DataFrame, categoria: str, n: int = 4,
                          seed: int = 42):
    subset = df[df["tipo"] == categoria]
    if subset.empty:
        print(f"  (nessun esempio per '{categoria}')")
        return
    subset = subset.sample(min(n, len(subset)), random_state=seed)

    fig, axes = plt.subplots(len(subset), 2, figsize=(7, 2.6 * len(subset)))
    if len(subset) == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(subset.iterrows()):
        try:
            img, mask = _open_tile(row["tile_path"])
        except Exception as e:
            axes[i, 0].set_title(f"ERR: {e}", fontsize=8, color="red")
            continue

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(
            os.path.basename(row["tile_path"]),
            fontsize=8,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="RdYlGn_r", vmin=0, vmax=1)
        annot = (
            f"bianco={row['white_percent']:.1f}% "
            f"block={row['max_block_frac']:.2f} "
            f"std={row['img_std']:.3f}\n"
            f"dark={row['dark_frac']:.2f} "
            f"fft={row['stripe_fft_score']:.1f} "
            f"align={row['align_score']:.2f}"
        )
        axes[i, 1].set_title(annot, fontsize=7)
        axes[i, 1].axis("off")

    fig.suptitle(f"Esempi — {categoria}",
                 fontsize=12, fontweight="bold", color=COLOR_MAP.get(categoria, "white"))
    plt.tight_layout()
    plt.show()


def visualizza_tutte(df: pd.DataFrame, n_per_categoria: int = 4):
    for cat in CATEGORIE_ORDINATE:
        if cat in df["tipo"].values:
            print(f"\n>>> Categoria: {cat}")
            visualizza_categoria(df, cat, n=n_per_categoria)


# ---------------------------------------------------------------------------
# Confronto v2 vs v3
# ---------------------------------------------------------------------------

def confronto_v2_v3(df: pd.DataFrame):
    """Riproduce velocemente la classificazione v2 e mostra dove diverge."""
    print("\n" + "=" * 60)
    print("  CONFRONTO v2 (notebook) vs v3 (questo script)")
    print("=" * 60)

    def v2(row):
        if row["white_percent"] >= 100.0:
            return "tutto_bianco"
        if (row["n_white_rows"] > 5 and row["n_transitions"] >= 2):
            return "artefatto_righe"
        if row["white_percent"] > 99.0 and row["n_white_rows"] > 200:
            return "artefatto_righe"
        return "normale"

    df["tipo_v2"] = df.apply(v2, axis=1)
    cross = pd.crosstab(df["tipo_v2"], df["tipo"])
    print(cross.to_string())

    smascherate = df[(df["tipo_v2"] == "normale") & (df["tipo"] != "normale")]
    print(
        f"\n  Tile che v2 classificava 'normale' ma v3 scarta: "
        f"{len(smascherate)} "
        f"({len(smascherate) / max(1, (df['tipo_v2'] == 'normale').sum()) * 100:.1f}%)"
    )
    if len(smascherate):
        print("  Suddivise per nuova categoria:")
        for cat, n in smascherate["tipo"].value_counts().items():
            print(f"    - {cat:20s}: {n}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Carico {INDEX_PATH}")
    df = pd.read_csv(INDEX_PATH)
    print(f"Tile totali: {len(df)}")

    df = analizza_dataset(df)
    stampa_distribuzione(df)

    df.to_csv(REPORT_PATH, index=False)
    df[df["tipo"] == "normale"].to_csv(CLEAN_PATH, index=False)
    df[df["tipo"] == "misaligned"].to_csv(RECOVER_PATH, index=False)
    print(f"\nSalvati:")
    print(f"  - {REPORT_PATH}")
    print(f"  - {CLEAN_PATH}")
    print(f"  - {RECOVER_PATH}")

    confronto_v2_v3(df)

    print("\nVisualizzo esempi per ogni categoria...")
    visualizza_tutte(df, n_per_categoria=4)


if __name__ == "__main__":
    main()
