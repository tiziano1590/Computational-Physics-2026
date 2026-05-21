"""
Curator interattivo per le tile classificate 'normale' dalla v4.

Apre una finestra matplotlib con:
  - immagine + maschera della tile corrente
  - 4 bottoni in basso: Indietro / Scarta / Tieni / Avanti
  - shortcut da tastiera (vedi sotto)

Le decisioni vengono salvate in tiles/curator_decisions.csv AD OGNI click.
Riprende automaticamente dalla prima tile senza decisione.

Uso:
    cd Moon-Recognition/notebooks
    python tile_curator.py

Shortcut da tastiera:
    ←   o   Backspace   →  Indietro (no decisione)
    D                   →  Scarta e avanti
    K                   →  Tieni e avanti
    →   o   Spazio      →  Avanti senza cambiare decisione
    Q   o   Esc         →  Chiudi
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'lunar_segmentation', 'data', 'MR')
REPORT_PATH = os.path.join(DATA_DIR, 'tiles', 'corruption_report_v4.csv')
DECISIONS_PATH = os.path.join(DATA_DIR, 'tiles', 'curator_decisions.csv')


def main():
    # --- Carica report v4 e filtra a 'normale' ---
    if not os.path.exists(REPORT_PATH):
        raise SystemExit(
            f"Manca {REPORT_PATH}. Esegui prima il notebook v4 per generarlo."
        )
    df = pd.read_csv(REPORT_PATH)
    candidates = df[df['tipo'] == 'normale'].reset_index(drop=True)
    N = len(candidates)
    print(f"Tile da curare: {N}")

    # --- Decisioni precedenti se esistono ---
    if os.path.exists(DECISIONS_PATH):
        decisions = (pd.read_csv(DECISIONS_PATH)
                       .set_index('tile_path')['decision']
                       .to_dict())
        print(f"Caricate {len(decisions)} decisioni da {DECISIONS_PATH}")
    else:
        decisions = {}

    # Punto di ripartenza = prima tile senza decisione
    state = {'idx': 0}
    for i, row in candidates.iterrows():
        if row['tile_path'] not in decisions:
            state['idx'] = int(i)
            break

    # --- Setup figura ---
    plt.rcParams['figure.facecolor'] = '#1a1a2e'
    plt.rcParams['axes.facecolor']   = '#16213e'
    plt.rcParams['text.color']       = '#e0e0e0'
    plt.rcParams['axes.labelcolor']  = '#e0e0e0'
    plt.rcParams['xtick.color']      = '#a0a0a0'
    plt.rcParams['ytick.color']      = '#a0a0a0'

    fig = plt.figure(figsize=(12, 6.5))
    fig.canvas.manager.set_window_title('Tile Curator — v4')

    # Aree immagine + maschera (sopra) e bottoni (sotto)
    ax_img  = fig.add_axes([0.04, 0.18, 0.45, 0.74])
    ax_mask = fig.add_axes([0.52, 0.18, 0.45, 0.74])

    ax_back = fig.add_axes([0.06, 0.04, 0.20, 0.07])
    ax_drop = fig.add_axes([0.28, 0.04, 0.20, 0.07])
    ax_keep = fig.add_axes([0.52, 0.04, 0.20, 0.07])
    ax_skip = fig.add_axes([0.74, 0.04, 0.20, 0.07])

    btn_back = Button(ax_back, '← Indietro',  color='#34495e', hovercolor='#4a6178')
    btn_drop = Button(ax_drop, '✗ Scarta',    color='#c0392b', hovercolor='#e74c3c')
    btn_keep = Button(ax_keep, '✓ Tieni',     color='#27ae60', hovercolor='#2ecc71')
    btn_skip = Button(ax_skip, 'Avanti →',    color='#34495e', hovercolor='#4a6178')
    for btn in (btn_back, btn_drop, btn_keep, btn_skip):
        btn.label.set_color('white')
        btn.label.set_fontweight('bold')

    def save_decisions():
        """Riscrive l'intero CSV ad ogni click — istantaneo a queste dimensioni."""
        pd.DataFrame(
            [{'tile_path': k, 'decision': v} for k, v in decisions.items()]
        ).to_csv(DECISIONS_PATH, index=False)

    def render():
        idx = state['idx']
        row = candidates.iloc[idx]
        current = decisions.get(row['tile_path'], '—')

        # Carica tile
        data = np.load(os.path.join(DATA_DIR, row['tile_path']))
        img  = data['image'][2]  # Canale Sobel pre-calcolato
        mask = data['mask'][0]

        ax_img.clear()
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(
            f"{os.path.basename(row['tile_path'])} — Sobel",
            fontsize=11, color='white'
        )
        ax_img.axis('off')

        ax_mask.clear()
        ax_mask.imshow(mask, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax_mask.set_title(
            f"crateri={int(row['n_craters'])}   "
            f"align={row['align_score']:.2f}   "
            f"dark={row['dark_frac']:.2f}   "
            f"std={row['img_std']:.3f}",
            fontsize=10, color='white',
        )
        ax_mask.axis('off')

        # Titolo globale con stato decisione
        n_keep = sum(1 for v in decisions.values() if v == 'keep')
        n_drop = sum(1 for v in decisions.values() if v == 'discard')
        n_todo = N - n_keep - n_drop
        color = {'keep': '#2ecc71', 'discard': '#e74c3c'}.get(current, '#a0a0a0')

        fig.suptitle(
            f"Tile {idx+1} / {N}    |    Decisione corrente: [{current}]    |    "
            f"keep={n_keep}   discard={n_drop}   todo={n_todo}",
            color=color, fontsize=12, fontweight='bold',
        )
        fig.canvas.draw_idle()

    def go(delta):
        state['idx'] = max(0, min(N - 1, state['idx'] + delta))
        render()

    def on_back(_=None): go(-1)
    def on_skip(_=None): go(+1)
    def on_keep(_=None):
        decisions[candidates.iloc[state['idx']]['tile_path']] = 'keep'
        save_decisions()
        go(+1)
    def on_drop(_=None):
        decisions[candidates.iloc[state['idx']]['tile_path']] = 'discard'
        save_decisions()
        go(+1)

    btn_back.on_clicked(on_back)
    btn_drop.on_clicked(on_drop)
    btn_keep.on_clicked(on_keep)
    btn_skip.on_clicked(on_skip)

    # Shortcut da tastiera
    def on_key(event):
        k = event.key
        if k in ('left', 'backspace'):
            on_back()
        elif k in ('right', ' '):
            on_skip()
        elif k == 'k':
            on_keep()
        elif k == 'd':
            on_drop()
        elif k in ('q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    render()
    plt.show()


if __name__ == '__main__':
    main()
