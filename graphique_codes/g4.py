import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

# --- PALETTE ---
COLORS = {
    'bg':   '#0F0A1C',
    'cyan': '#18FFF6',
    'rose': '#FF2A6D',
    'text': '#E6D3FF',
    'grid': '#2A1B3D',
    'white': '#FFFFFF',
    'noise': '#6c687d',  # Un gris un peu plus violet pour s'intégrer
}

def neon(c, lw=2.0, alpha_glow=0.4):
    return [
        pe.Stroke(linewidth=lw*4, foreground=c, alpha=alpha_glow*0.35),
        pe.Stroke(linewidth=lw*1.5, foreground=c, alpha=alpha_glow),
        pe.Normal()
    ]

def softmax_stable(scores: np.ndarray, axis: int = 0) -> np.ndarray:
    """Softmax stable pour éviter les warnings overflow"""
    m = np.max(scores, axis=axis, keepdims=True)
    ex = np.exp(scores - m)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def y_at(x_grid: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Récupère la hauteur exacte de la courbe à x0"""
    idx = int(np.argmin(np.abs(x_grid - x0)))
    return float(y[idx])

def plot_attention_hijack_3channels_final():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Space Grotesk', 'Arial', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # --- 1) GRILLE ---
    x = np.linspace(0, 100, 700)

    # --- 2) SCORES (Logits) ---
    # SYSTEM : Pic initial fort, puis s'écrase vite
    scores_sys = 4.0 * np.exp(-0.12 * (x - 8)**2) + 0.1

    # ATTAQUE : Vagues successives qui saturent l'espace
    scores_att = np.zeros_like(x)
    for pos in range(25, 115, 10):
        scores_att += 7.0 * np.exp(-0.04 * (x - pos)**2)
    scores_att += 0.2

    # BRUIT : Le fond de cuve (tokens grammaticaux, formatage...)
    scores_noise = np.full_like(x, 0.5) # Un peu plus présent pour montrer la compétition

    # --- 3) SOFTMAX ---
    scores = np.vstack([scores_sys, scores_att, scores_noise])
    alpha = softmax_stable(scores, axis=0)
    alpha_sys, alpha_att, alpha_noise = alpha

    # --- 4) COURBES ---
    
    # A. Bruit (Fond) - Dessiné en premier
    ax.plot(x, alpha_noise, color=COLORS['noise'], lw=1.5, alpha=0.6, label='Bruit')
    ax.fill_between(x, 0, alpha_noise, color=COLORS['noise'], alpha=0.1)

    # B. Système (Cyan) - Le héros malheureux
    ax.plot(x, alpha_sys, color=COLORS['cyan'], lw=2.5, path_effects=neon(COLORS['cyan']))
    ax.fill_between(x, 0, alpha_sys, color=COLORS['cyan'], alpha=0.2)

    # C. Attaque (Rose) - Le prédateur
    ax.plot(x, alpha_att, color=COLORS['rose'], lw=2.5, path_effects=neon(COLORS['rose']))
    ax.fill_between(x, 0, alpha_att, color=COLORS['rose'], alpha=0.2)

    # --- 5) BLOCS MÉMOIRE (Bas) ---
    y_base = -0.15
    h_block = 0.10

    # Bloc System
    rect_sys = Rectangle((0, y_base), 20, h_block, color=COLORS['cyan'], alpha=0.9, zorder=5)
    ax.add_patch(rect_sys)
    ax.text(10, y_base + h_block/2, "SYSTEM PROMPT", color='black',
            ha='center', va='center', fontsize=9, fontweight='bold', zorder=20,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))

    # Bloc User
    rect_att = Rectangle((20, y_base), 80, h_block, color=COLORS['rose'], alpha=0.9, zorder=5)
    ax.add_patch(rect_att)
    ax.text(60, y_base + h_block/2, "USER INPUT / NOISE FLOOD", color='black',
            ha='center', va='center', fontsize=9, fontweight='bold', zorder=20,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))

    # Flèche Continuité
    ax.annotate("", xy=(99, y_base - 0.03), xytext=(1, y_base - 0.03),
                arrowprops=dict(arrowstyle="->", color='white', lw=1.5, alpha=0.5))
    ax.text(50, y_base - 0.08, "FLUX UNIQUE (MÉMOIRE PARTAGÉE CONTINUE)",
            color='white', ha='center', fontsize=8, fontweight='bold', alpha=0.7)

    # --- 6) ANNOTATIONS AMÉLIORÉES ---
    
    # 1. Effondrement : On vise le croisement
    crush_x = 22
    crush_y = y_at(x, alpha_sys, crush_x)
    ax.annotate(r"EFFONDREMENT" + "\n" + r"($v_{s\acute{e}cu}$ noyé)",
                xy=(crush_x, crush_y), xytext=(crush_x + 5, 0.60),
                arrowprops=dict(arrowstyle="->", color=COLORS['cyan']),
                color=COLORS['cyan'], fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS['bg'], ec=COLORS['cyan']))

    # 2. Capture : On vise un sommet
    plateau_x = 55
    plateau_y = y_at(x, alpha_att, plateau_x)
    ax.annotate("CAPTURE CONTINUE\n" + r"($v_{adv}$ dominant)",
                xy=(plateau_x, plateau_y), xytext=(plateau_x, 1.15),
                arrowprops=dict(arrowstyle="->", color=COLORS['rose']),
                color=COLORS['rose'], fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS['bg'], ec=COLORS['rose']))

    # 3. Label Bruit (plus visible)
    ax.text(2, 0.15, r"$\epsilon$ (bruit structurel)", color=COLORS['noise'], fontsize=9, alpha=0.9, fontweight='bold')

    # 4. Équation Vectorielle (Déplacée pour lisibilité)
    # On la met dans un encart sombre pour qu'elle soit lisible par dessus la courbe rose
    ax.text(78, 0.85, r"$v_{\mathrm{final}} \approx \epsilon + v_{\mathrm{adv}} + v_{\mathrm{s\acute{e}cu}}$",
            color=COLORS['text'], fontsize=11, zorder=30,
            bbox=dict(boxstyle="round,pad=0.4", fc=COLORS['bg'], ec=COLORS['grid'], alpha=0.9))

    # --- 7) TITRES & LAYOUT ---
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.35, 1.35) # Un peu plus de marge en haut
    
    ax.text(0, 1.08, "CONFUSION DES PLANS : LE MONISME ARCHITECTURAL",
            transform=ax.transAxes, color=COLORS['cyan'], fontsize=16, fontweight='bold',
            path_effects=neon(COLORS['cyan'], 0.5))

    ax.text(0, 1.02, "VISUALISATION SOFTMAX : L'ATTAQUE CAPTURE LA MASSE, LE RESTE DEVIENT RÉSIDUEL",
            transform=ax.transAxes, color='white', fontsize=10, alpha=0.85)

    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_attention_hijack_3channels_final()