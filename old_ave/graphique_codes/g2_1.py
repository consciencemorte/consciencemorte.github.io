import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import matplotlib.font_manager as fm

# --- PALETTE CYBERPUNK ---
COLORS = {
    "bg":      "#0F0A1C",
    "stream":  "#E6D3FF",  # Flux violet pâle
    "core":    "#FFFFFF",  # Coeur blanc du flux (énergie)
    "mha":     "#18FFF6",  # Cyan
    "mlp":     "#FF2A6D",  # Rose
    "add":     "#FFD700",  # Or
    "text":    "#FFFFFF",
    "muted":   "#A59AC2",
    "panel":   (0.06, 0.06, 0.12, 0.85),
}

def pick_font(candidates=("Space Grotesk", "Arial", "Inter", "DejaVu Sans")):
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in avail: return c
    return "DejaVu Sans"

def glow(color, lw, alpha=0.3):
    return [pe.Stroke(linewidth=lw*3, foreground=color, alpha=alpha), pe.Normal()]

# Reworked draw_stream_segment pour un épaississement plus agressif (X1.8)
def draw_stream_segment(ax, x0, x1, y, lw_factor, alpha):
    # Les multiplicateurs sont augmentés pour un contraste fort
    lw_stream = lw_factor * 1.8 
    lw_core = lw_factor * 0.35 
    
    # 1. Le halo (Glow)
    ax.plot([x0, x1], [y, y], color=COLORS["stream"], lw=lw_stream*2.5, alpha=alpha*0.18, 
            solid_capstyle="butt", zorder=1)
    
    # 2. Le corps du flux (Violet)
    ax.plot([x0, x1], [y, y], color=COLORS["stream"], lw=lw_stream, alpha=alpha, 
            solid_capstyle="butt", zorder=2)
    
    # 3. Le coeur d'énergie (Blanc fin au centre)
    ax.plot([x0, x1], [y, y], color=COLORS["core"], lw=lw_core, alpha=0.7, 
            solid_capstyle="butt", zorder=3)


def add_box(ax, cx, cy, w, h, edge, title, subtitle=None):
    rect = patches.FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.2",
        fc=COLORS["panel"], ec=edge, lw=1.5,
        path_effects=glow(edge, 1.5, 0.2), zorder=10
    )
    ax.add_patch(rect)
    
    ax.text(cx, cy+0.15, title, ha="center", va="center", 
            color=edge, fontsize=10, fontweight="bold", zorder=11)
    
    if subtitle:
        ax.text(cx, cy-0.15, subtitle, ha="center", va="center", 
                color=COLORS["text"], fontsize=8.5, alpha=0.9, zorder=11)

def plus_node(ax, x, y, color, size=200):
    ax.scatter(x, y, s=size, c=COLORS["bg"], zorder=19)
    ax.scatter(x, y, s=size, c=COLORS["bg"], edgecolors=color, lw=2, zorder=20,
               path_effects=glow(color, 2))
    ax.text(x, y, "+", color=color, ha="center", va="center", 
            fontsize=14, fontweight="bold", zorder=21)

def curved_arrow(ax, p1, p2, color, rad=0.3, ls="-", lw=2):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        lw=lw, linestyle=ls, color=color,
        mutation_scale=15, zorder=15,
        path_effects=glow(color, lw) if ls=="-" else None
    )
    ax.add_patch(arr)

def plot_residual_highway_final():
    font = pick_font()
    plt.rcParams.update({"font.family": font, "mathtext.fontset": "dejavusans"})
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Configuration géométrique
    y_stream = 0
    layers = 3
    spacing = 5.0
    start_x = 0
    box_w, box_h = 2.2, 1.0
    
    # Rôles fonctionnels
    roles_mha = ["Copie locale\n(Syntaxe)", "Induction Heads\n(Patterns)", "Routage Global\n(Sémantique)"]
    roles_mlp = ["Faits simples", "Mémoire\nAssociative", "Raffinement\n(Output)"]

    # --- INPUT ---
    ax.text(start_x, y_stream + 0.35, "INPUT", color=COLORS["stream"], ha="left", fontweight="bold")
    ax.text(start_x, y_stream - 0.35, r"$x_0$", color=COLORS["stream"], ha="left", fontsize=14)
    ax.annotate("", xy=(start_x, y_stream), xytext=(start_x-1.5, y_stream),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["stream"], lw=3, mutation_scale=20))

    # --- BOUCLE PRINCIPALE ---
    curr_x = start_x
    base_lw_factor = 3.0 
    step_lw_factor = 1.6 
    
    for i in range(layers):
        next_x = curr_x + spacing
        
        # Facteur de ligne de base pour cette étape
        lw_factor = base_lw_factor + i * step_lw_factor 
        
        # Position des sous-blocs
        x_mha = curr_x + spacing * 0.3
        x_mlp = curr_x + spacing * 0.7
        
        # 1. Segment : Début -> MHA
        draw_stream_segment(ax, curr_x, x_mha, y_stream, lw_factor, alpha=0.5)
        
        # --- BLOC MHA (Haut) ---
        y_mha = 1.8
        add_box(ax, x_mha, y_mha, box_w, box_h, COLORS["mha"], "ATTENTION", roles_mha[i])
        
        # Lecture (Pointillés)
        curved_arrow(ax, (x_mha - 0.5, y_stream + 0.1), (x_mha - 0.8, y_mha), 
                     COLORS["mha"], rad=0.2, ls=":", lw=1.5)
        # Écriture (Pleine)
        curved_arrow(ax, (x_mha + 0.8, y_mha), (x_mha, y_stream + 0.1), 
                     COLORS["mha"], rad=-0.2, ls="-", lw=2.5)
        
        plus_node(ax, x_mha, y_stream, COLORS["mha"])

        # 2. Segment : MHA -> MLP
        lw_factor_mha_out = lw_factor + step_lw_factor * 0.5
        draw_stream_segment(ax, x_mha, x_mlp, y_stream, lw_factor_mha_out, alpha=0.55)

        # --- BLOC MLP (Bas) ---
        y_mlp = -1.8
        add_box(ax, x_mlp, y_mlp, box_w, box_h, COLORS["mlp"], "MLP", roles_mlp[i])
        
        # Lecture
        curved_arrow(ax, (x_mlp - 0.5, y_stream - 0.1), (x_mlp - 0.8, y_mlp), 
                     COLORS["mlp"], rad=-0.2, ls=":", lw=1.5)
        # Écriture
        curved_arrow(ax, (x_mlp + 0.8, y_mlp), (x_mlp, y_stream - 0.1), 
                     COLORS["mlp"], rad=0.2, ls="-", lw=2.5)
        
        plus_node(ax, x_mlp, y_stream, COLORS["mlp"])

        # 3. Segment : MLP -> Fin de couche
        lw_factor_mlp_out = lw_factor_mha_out + step_lw_factor * 0.5
        draw_stream_segment(ax, x_mlp, next_x, y_stream, lw_factor_mlp_out, alpha=0.6)
        
        # Label Couche
        ax.text((x_mha + x_mlp)/2, 0.5, f"COUCHE {i+1}", color=COLORS["muted"], 
                ha="center", fontsize=8, alpha=0.7)

        curr_x = next_x

    # --- OUTPUT ---
    final_lw_factor = lw_factor_mlp_out if 'lw_factor_mlp_out' in locals() else base_lw_factor
    
    # Flèche de sortie
    ax.annotate("", xy=(curr_x + 1.5, y_stream), xytext=(curr_x, y_stream),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["stream"], 
                                lw=final_lw_factor * 1.8, # Utiliser le multiplicateur du flux pour la flèche
                                mutation_scale=25))
    
    ax.text(curr_x + 1.2, y_stream + 0.35, "OUTPUT", color=COLORS["stream"], ha="right", fontweight="bold")
    ax.text(curr_x + 1.2, y_stream - 0.35, r"$x_L$", color=COLORS["stream"], ha="right", fontsize=14)

    # --- TEXTES GLOBAUX ---
    
    # Formule (Déplacée à y=-3.0)
    bbox_eq = dict(boxstyle="round,pad=0.6", fc=(0,0,0,0.3), ec=COLORS["stream"], lw=1)
    ax.text(start_x + (spacing*layers)/2, -3.0, # NOUVELLE POSITION
            r"$x_L = x_0 + \sum \Delta_{Att} + \sum \Delta_{MLP}$", 
            color=COLORS["text"], fontsize=14, ha="center", bbox=bbox_eq)

    # Titre (En haut)
    ax.text(start_x + (spacing*layers)/2, 2.8, 
            "MÉMOIRE RÉSIDUELLE : L'INFORMATION S'ACCUMULE SANS S'EFFACER", 
            color=COLORS["add"], fontsize=12, fontweight="bold", ha="center",
            path_effects=glow(COLORS["add"], 2.2, 0.22))

    # Nettoyage
    ax.set_xlim(start_x - 1.5, curr_x + 1.6)
    ax.set_ylim(-3.5, 3.5)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_residual_highway_final()