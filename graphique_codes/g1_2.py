import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# --- PALETTE ---
COLORS = {
    'bg':   '#0F0A1C',
    'cyan': '#18FFF6',
    'rose': '#F5A3B5',
    'vio':  '#7F3FE4',
    'text': '#E6D3FF',
}

def neon(c, lw=2.0, alpha_glow=0.25):
    return [
        pe.Stroke(linewidth=lw*3.2, foreground=c, alpha=alpha_glow*0.45),
        pe.Stroke(linewidth=lw*1.6, foreground=c, alpha=alpha_glow),
        pe.Normal()
    ]

def wedge(ax, ang1, ang2, r, color, alpha=0.12, z=1):
    th = np.linspace(ang1, ang2, 220)
    xs = np.r_[0, r*np.cos(th), 0]
    ys = np.r_[0, r*np.sin(th), 0]
    ax.fill(xs, ys, color=color, alpha=alpha, ec=None, zorder=z)

def draw_arrow(ax, start, end, color, lw=2.6, ms=18, ls='-', z=10, glow=True, alpha=1.0):
    pe_list = neon(color, lw, 0.22) if glow else []
    arr = FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=ms,
        linewidth=lw, linestyle=ls, color=color, alpha=alpha,
        joinstyle='round', capstyle='round', zorder=z
    )
    if pe_list:
        arr.set_path_effects(pe_list)
    ax.add_patch(arr)

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def place_text_under_segment(ax, A, B, text, color, t=0.55, offset=-0.55,
                             fontsize=12, weight='bold', alpha=0.9, glow=True, z=25):
    """Place un texte à t% le long du segment AB, décalé perpendiculairement."""
    A = np.array(A, float); B = np.array(B, float)
    v = B - A
    if np.linalg.norm(v) < 1e-9:
        return
    p = A + t * v
    n = unit(np.array([-v[1], v[0]]))  # normale
    p = p + offset * n

    effects = neon(color, 0.6) if glow else None
    ax.text(p[0], p[1], text, color=color, fontsize=fontsize, fontweight=weight,
            ha='center', va='center', alpha=alpha, zorder=z,
            path_effects=effects if effects else None)

def plot_final_cyberpunk_fixed():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Space Grotesk', 'Arial', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    O = np.array([0., 0.])

    # --- Données ---
    v_secu = np.array([-1.5, 3.0])
    v_adv  = np.array([9.5, 0.8])
    x      = v_secu + v_adv

    R = 3.5
    x_tilde = unit(x) * R

    # --- 1) ZONES (renforcées + label) ---
    rmax = 12
    wedge(ax, np.deg2rad(100), np.deg2rad(160), rmax, COLORS['cyan'], alpha=0.09, z=1)  # refus
    wedge(ax, np.deg2rad(-10),  np.deg2rad(30),  rmax, COLORS['rose'], alpha=0.10, z=2)  # complaisance (au-dessus)

    ax.text(6.2, 1.9, "Zone complaisance", color=COLORS['rose'],
            fontsize=11, fontweight='bold', alpha=0.75, zorder=30)

    # --- 2) VECTEURS ---
    # Sécurité
    draw_arrow(ax, O, v_secu, COLORS['cyan'], lw=2.8, ms=20, z=15)
    ax.text(v_secu[0]-0.25, v_secu[1]+0.45, "Alignement\n(Sécurité)",
            color=COLORS['cyan'], fontsize=10, fontweight='bold', ha='right',
            path_effects=neon(COLORS['cyan'], 0.6), zorder=25)

    # Attaque (pointillée)
    draw_arrow(ax, v_secu, x, COLORS['rose'], lw=2.8, ms=20, ls='--', z=16)

    mid_adv = v_secu + 0.5 * v_adv
    ax.text(mid_adv[0], mid_adv[1]+0.65, "Vecteur adversarial\n(Saturation)",
            color=COLORS['rose'], fontsize=10, fontweight='bold', ha='center', rotation=4,
            path_effects=neon(COLORS['rose'], 0.6), zorder=25)

    # Résultante (ligne violette)
    ax.plot([O[0], x[0]], [O[1], x[1]],
            color=COLORS['vio'], linestyle=':', linewidth=1.6, alpha=0.8, zorder=6)

    # --- 3) RMSNorm ---
    circle = plt.Circle(O, R, color=COLORS['text'], fill=False,
                        linestyle='--', linewidth=1.1, alpha=0.28, zorder=3)
    ax.add_patch(circle)

    ax.scatter(*x_tilde, color=COLORS['bg'], edgecolors=COLORS['vio'],
               s=140, zorder=20, linewidth=2)
    ax.scatter(*x_tilde, color=COLORS['vio'], s=55, zorder=21)
    ax.text(x_tilde[0]-0.9, x_tilde[1]+0.55, r"$\tilde{x}$ (Normalisé)",
            color=COLORS['vio'], fontsize=11, fontweight='bold', zorder=25)

    ax.scatter(*x, color=COLORS['bg'], edgecolors=COLORS['vio'],
               s=80, zorder=10, alpha=0.55, linewidth=1.2)
    ax.text(x[0]+0.25, x[1], r"$x$ (Total)", color=COLORS['vio'],
            fontsize=11, alpha=0.7, zorder=25)

    # --- 4) Inégalité placée automatiquement sous la flèche rose ---
    place_text_under_segment(
        ax, v_secu, x,
        r"$\|v_{adv}\| \gg \|v_{secu}\|$",
        COLORS['rose'],
        t=0.58, offset=-0.60, fontsize=12, alpha=0.9
    )

    # Titres
    ax.text(-3.8, 6.5, "GÉOMÉTRIE DE LA SATURATION", color=COLORS['cyan'],
            fontsize=16, fontweight='bold', path_effects=neon(COLORS['cyan'], 0.6), zorder=30)
    ax.text(-3.8, 5.9, "L'AMPLITUDE DICTE LA DIRECTION", color=COLORS['text'],
            fontsize=11, alpha=0.8, zorder=30)

    # Boîte résultat
    bbox = dict(boxstyle="round,pad=0.6", fc=COLORS['bg'], ec=COLORS['vio'], alpha=0.9, lw=1.5)
    ax.text(2.5, -2.2,
            " RÉSULTAT :\n La norme écrase l'angle de sécurité.\n La direction devient adversariale.",
            color=COLORS['text'], fontsize=10, bbox=bbox, family='monospace', zorder=30)

    ax.set_xlim(-4, 11)
    ax.set_ylim(-3, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_final_cyberpunk_fixed()
