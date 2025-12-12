import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import matplotlib.patheffects as pe

# --- PALETTE ---
COLORS = {
    'bg':   '#0F0A1C',
    'cyan': '#18FFF6',
    'rose': '#FF2A6D', 
    'vio':  '#D355FF', 
    'text': '#E6D3FF',
    'grid': '#2A1B3D'
}

def neon(c, lw=2.0, alpha_glow=0.4):
    return [
        pe.Stroke(linewidth=lw*4, foreground=c, alpha=alpha_glow*0.4),
        pe.Stroke(linewidth=lw*1.5, foreground=c, alpha=alpha_glow),
        pe.Normal()
    ]

def plot_fragmentation_hud_final():
    # Setup
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Space Grotesk', 'Arial', 'DejaVu Sans', 'sans-serif']
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # --- Données ---
    center = np.array([0.5, 0.5])
    
    # Fragments
    f1 = center + np.array([-0.30, -0.20]) # "Mal"
    f2 = center + np.array([ 0.25, -0.25]) # "is"
    f3 = center + np.array([ 0.15,  0.35]) # "cious"
    fragments = np.array([f1, f2, f3])
    
    # Barycentre
    reconstruction = fragments.mean(axis=0) 
    
    # Token unique (Bloqué) 
    token_blocked = reconstruction + np.array([-0.04, 0.02]) 

    # --- 1. Fond ---
    ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=1, alpha=0.3)
    
    # --- 2. Zone Interdite ---
    for r, alpha in zip([0.15, 0.25, 0.35], [0.3, 0.2, 0.1]):
        circ = Circle(center, r, color=COLORS['rose'], fill=True, alpha=alpha*0.3, zorder=1)
        ax.add_patch(circ)
        circ_line = Circle(center, r, color=COLORS['rose'], fill=False, ls='--', lw=1, alpha=0.5, zorder=1)
        ax.add_patch(circ_line)

    # --- 2b. Grille polaire dans la zone interdite ---
    num_radial = 12
    num_circles = 6

    angles = np.linspace(0, 2*np.pi, num_radial, endpoint=False)
    radii  = np.linspace(0.05, 0.35, num_circles)

    # Lignes radiales
    for theta in angles:
        x = center[0] + radii * np.cos(theta)
        y = center[1] + radii * np.sin(theta)
        ax.plot(x, y, color=COLORS['rose'], lw=0.6, alpha=0.10, zorder=1)

    # Cercles internes
    for r in radii:
        circle_grid = Circle(center, r, fill=False,
                             edgecolor=COLORS['rose'], lw=0.6,
                             alpha=0.10, zorder=1)
        ax.add_patch(circle_grid)

    ax.text(center[0], center[1]+0.38, "ZONE SÉMANTIQUE INTERDITE", 
            color=COLORS['rose'], fontsize=10, ha='center', fontweight='bold')

    # --- 3. Triangle ---
    poly = Polygon(fragments, closed=True, color=COLORS['cyan'], fill=False, 
                   ls=':', lw=1.5, alpha=0.6, path_effects=neon(COLORS['cyan'], 1))
    ax.add_patch(poly)

    for frag in fragments:
        ax.plot([frag[0], reconstruction[0]], [frag[1], reconstruction[1]], 
                color=COLORS['cyan'], lw=0.8, alpha=0.3, zorder=2)

    # --- 4. Points & Labels ---
    # A. Les Fragments
    labels = ['"Mal"', '"is"', '"cious"']
    offsets = [(-0.02, -0.05), (0.02, -0.05), (0.0, 0.05)]
    
    for i, frag in enumerate(fragments):
        ax.scatter(*frag, c=COLORS['bg'], s=180, marker='^', edgecolors=COLORS['cyan'], 
                   lw=2, zorder=10, path_effects=neon(COLORS['cyan']))
        ax.text(frag[0]+offsets[i][0], frag[1]+offsets[i][1], labels[i], 
                color=COLORS['cyan'], fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc=COLORS['bg'], ec='none', alpha=0.6))

    # B. Le Token Bloqué
    ax.scatter(*token_blocked, c=COLORS['rose'], s=250, marker='X', zorder=15, 
               path_effects=neon(COLORS['rose']))
    
    ax.annotate("TOKEN UNIQUE\n[Malicious]\n[!] BLOQUÉ", 
                xy=token_blocked, xytext=(token_blocked[0]-0.25, token_blocked[1]),
                arrowprops=dict(arrowstyle="-", color=COLORS['rose'], lw=1),
                color=COLORS['rose'], fontsize=9, fontweight='bold', ha='right', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS['bg'], ec=COLORS['rose'], alpha=0.8))

    # C. La Reconstruction
    ax.scatter(*reconstruction, c=COLORS['bg'], s=300, marker='*', edgecolors=COLORS['vio'], 
               lw=2, zorder=20, path_effects=neon(COLORS['vio'], 3))
    ax.scatter(*reconstruction, c=COLORS['vio'], s=100, marker='*', zorder=21)

    ax.annotate("RECONSTRUCTION\n(Barycentre)\n INVISIBLE", 
                xy=reconstruction, xytext=(reconstruction[0]+0.25, reconstruction[1]),
                arrowprops=dict(arrowstyle="-", color=COLORS['vio'], lw=1),
                color=COLORS['vio'], fontsize=10, fontweight='bold', ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.4", fc=COLORS['bg'], ec=COLORS['vio'], alpha=0.9))

    # --- Titre ---
    ax.text(0.1, 0.95, "L'ILLUSION DE LA FRAGMENTATION", color=COLORS['cyan'], 
            fontsize=18, fontweight='bold', transform=ax.transAxes, path_effects=neon(COLORS['cyan'], 0.5))

    # Nettoyage
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 1.0)
    ax.axis('off') 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_fragmentation_hud_final()
