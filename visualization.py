import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation

def show_animation(frames: list):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
    plt.close()
    HTML(ani.to_jshtml())


def draw_rules(ax, exp, step):

    COLOR_L = "#4fc3f7"
    COLOR_R = "#ef5350"
    COLOR_COND = "#b0bec5"
    COLOR_VAL  = "#ffe082"

    ax.clear()
    ax.axis("off")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    action = exp["action"]
    rules  = exp["rules"]
    color  = COLOR_L if "Izquierda" in action else COLOR_R

    # Cabecera
    ax.text(0.05, 0.93, f"Paso {step}", fontsize=9,
            color="#546e7a", fontfamily="monospace", va="top")
    ax.text(0.05, 0.86, "ACCIÓN TOMADA", fontsize=7.5,
            color="#546e7a", fontfamily="monospace", va="top")
    ax.text(0.05, 0.79, action, fontsize=15, fontweight="bold",
            color=color, fontfamily="monospace", va="top")

    # Separador
    ax.axhline(0.73, xmin=0.04, xmax=0.96, color="#263238", linewidth=1)

    ax.text(0.05, 0.69, "Rama del árbol surrogado",
            fontsize=7, color="#546e7a", fontfamily="monospace", va="top")

    # Reglas
    y = 0.61
    for i, (feat, direction, thr, val) in enumerate(rules):
        # Flecha de rama
        prefix = "└─" if i == len(rules) - 1 else "├─"
        ax.text(0.05, y, prefix, fontsize=8, color="#37474f",
                fontfamily="monospace", va="top")
        # Feature name
        ax.text(0.14, y, feat, fontsize=8.5, color="#80cbc4",
                fontfamily="monospace", va="top", fontweight="bold")
        # Condición
        ax.text(0.14 + len(feat)*0.028 + 0.01, y, f" {direction} {thr:.3f}",
                fontsize=8.5, color=COLOR_COND, fontfamily="monospace", va="top")
        # Valor actual
        ax.text(0.75, y, f"[{val:.3f}]", fontsize=8, color=COLOR_VAL,
                fontfamily="monospace", va="top")
        y -= 0.09
        if y < 0.05:
            break

    # Nota pie
    ax.text(0.05, 0.04, "surrogate local · DQN CartPole-v1",
            fontsize=6.5, color="#263238", fontfamily="monospace", va="bottom")

def show_animation_with_decision_path(frames: list, explanations: list):
    fig = plt.figure(figsize=(13, 4.5), facecolor="#0f1117")

    ax_env = fig.add_axes([0.0, 0.0, 0.45, 1.0])
    ax_env.axis("off")

    ax_rules = fig.add_axes([0.47, 0.0, 0.53, 1.0])
    ax_rules.axis("off")
    ax_rules.set_facecolor("#0f1117")

    im = ax_env.imshow(frames[0])

    draw_rules(ax_rules, explanations[0], 0)

    def _update(i):
        im.set_data(frames[i])
        draw_rules(ax_rules, explanations[i], i)
        return []

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=50, blit=False)
    plt.close()
    return ani