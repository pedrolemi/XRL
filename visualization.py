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
