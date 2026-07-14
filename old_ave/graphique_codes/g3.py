import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# --- PALETTE CYBERPUNK ---
COLORS = {
    'bg':   '#0F0A1C',
    'cyan': '#18FFF6', 
    'rose': '#FF2A6D', 
    'text': '#E6D3FF',
    'grid': '#2A1B3D',
    'zone_s': '#18FFF6', 
    'zone_i': '#D355FF', 
    'zone_c': '#7F3FE4', 
    'zone_r': '#FF2A6D'  
}

def neon(c, lw=2.0, alpha_glow=0.4):
    return [
        pe.Stroke(linewidth=lw*4, foreground=c, alpha=alpha_glow*0.4),
        pe.Stroke(linewidth=lw*1.5, foreground=c, alpha=alpha_glow),
        pe.Normal()
    ]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)

def plot_logit_lens_scientific():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Space Grotesk', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    layers = np.arange(0, 33)

    # --- AJUSTEMENT DES PARAMÈTRES (PLUS ORGANIQUE) ---
    
    # 1. Attaque ("Sure") : Monte vite (réflexe) mais plafonne
    # Centre à 8, pente moyenne
    raw_sure = -4 + 14 / (1 + np.exp(-(0.35 * (layers - 8))))
    
    # 2. Défense ("Sorry") : Activation progressive dès la couche 22 (Induction de sécurité)
    # Au lieu d'un mur à 29, on commence à 24 avec une pente plus douce
    # Cela montre que le modèle "hésite" plus longtemps
    raw_sorry = -6 + 15 / (1 + np.exp(-(0.5 * (layers - 26))))
    
    # 3. Bruit
    raw_others = np.linspace(-1, -6, len(layers))

    # Softmax
    all_logits = np.vstack([raw_sure, raw_sorry, raw_others])
    all_probs = softmax(all_logits)
    prob_sure = all_probs[0]
    prob_sorry = all_probs[1]

    # --- ZONES ---
    # Ajustement des zones pour correspondre à la nouvelle dynamique
    ax.axvspan(0, 5, color=COLORS['zone_s'], alpha=0.03)
    ax.text(2.5, 1.08, "I. SURFACE", color=COLORS['zone_s'], ha='center', fontsize=8, fontweight='bold', alpha=0.8)

    ax.axvspan(5, 18, color=COLORS['zone_i'], alpha=0.03)
    ax.text(11.5, 1.08, "II. INDUCTION", color=COLORS['zone_i'], ha='center', fontsize=8, fontweight='bold', alpha=0.8)
            
    ax.axvspan(18, 25, color=COLORS['zone_c'], alpha=0.03) # Zone de confiance raccourcie
    ax.text(21.5, 1.08, "III. CONFIANCE", color=COLORS['zone_c'], ha='center', fontsize=8, fontweight='bold', alpha=0.8)

    # La zone de conflit commence plus tôt (couche 25)
    ax.axvspan(25, 32, color=COLORS['rose'], alpha=0.08)
    ax.text(28.5, 1.08, "IV. CONFLIT (RLHF)", color=COLORS['rose'], ha='center', fontsize=8, fontweight='bold', alpha=1.0)

    # --- TRACÉ ---
    ax.plot(layers, prob_sure, color=COLORS['cyan'], lw=3, label='Token: "Sure"', path_effects=neon(COLORS['cyan'], 1.5))
    ax.fill_between(layers, prob_sure, color=COLORS['cyan'], alpha=0.05)

    ax.plot(layers, prob_sorry, color=COLORS['rose'], lw=3, ls='--', label='Token: "Sorry"', path_effects=neon(COLORS['rose'], 2.0))

    # --- ANNOTATIONS ---
    
    # 1. Jailbreak
    ax.annotate("Réflexe d'instruction\n(Pre-training)", xy=(10, prob_sure[10]), xytext=(12, 0.45),
                arrowprops=dict(arrowstyle="->", color=COLORS['cyan']),
                color=COLORS['cyan'], fontsize=9, fontweight='bold')

    # 2. La Tension (Plus visible ici)
    # On voit bien que la courbe cyan baisse dès que la rose monte
    ax.annotate("", xy=(30, prob_sure[30]), xytext=(24, prob_sure[24]),
                arrowprops=dict(arrowstyle="->", color='white', ls=':', lw=1.5))
    
    ax.text(27, 0.65, "TENSION\nSTRUCTURELLE", color='white', fontsize=9, ha='center', alpha=0.9, fontweight='bold')

    # 3. Le Fail
    ax.annotate("Activation tardive\n(Safety Filter)", 
                xy=(32, prob_sorry[32]), xytext=(25, 0.2),
                arrowprops=dict(arrowstyle="->", color=COLORS['rose']),
                color=COLORS['rose'], fontsize=9, fontweight='bold')

    # --- SETUP ---
    ax.set_ylim(-0.02, 1.15)
    ax.set_xlim(0, 32)
    ax.grid(True, which='major', color=COLORS['grid'], linestyle='-', linewidth=1, alpha=0.5)
    ax.minorticks_on()
    
    ax.set_xlabel("Profondeur du Modèle (Couches)", color=COLORS['text'], fontsize=10)
    ax.set_ylabel("Probabilité (Softmax)", color=COLORS['text'], fontsize=10)
    
    ax.set_title("LOGIT LENS : DYNAMIQUE DE CONFLIT NEURONAL", color=COLORS['cyan'], 
                 fontsize=16, fontweight='bold', pad=25, path_effects=neon(COLORS['cyan'], 0.5))

    legend = ax.legend(loc='center left', facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_logit_lens_scientific()