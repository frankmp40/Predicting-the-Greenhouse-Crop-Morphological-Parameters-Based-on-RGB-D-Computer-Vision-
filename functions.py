from matplotlib import pyplot as plt


def plot_relationship(x, y, fig_label, X_label):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y['LeafArea'], linestyle='', marker='x')
    axs[0, 0].set_ylabel('LeafArea cm2')
    axs[0, 0].set_xlabel(X_label)
    axs[0, 1].plot(x, y['Diameter'], linestyle='', marker='x')
    axs[0, 1].set_ylabel('Diameter cm')
    axs[0, 1].set_xlabel(X_label)
    axs[1, 0].plot(x, y['Height'], linestyle='', marker='x')
    axs[1, 0].set_ylabel('Height cm')
    axs[1, 0].set_xlabel(X_label)
    axs[1, 1].plot(x, y['DMC'], linestyle='', marker='x')
    axs[1, 1].set_ylabel('DMC %')
    axs[1, 1].set_xlabel(X_label)

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # left  = 0.125  # the left side of the subplots of the figure
    # right = 0.9    # the right side of the subplots of the figure
    # bottom = 0.1   # the bottom of the subplots of the figure
    # top = 0.9      # the top of the subplots of the figure
    # wspace = 0.2   # the amount of width reserved for blank space between subplots
    # hspace = 0.2   # the amount of height reserved for white space between subplots

    fig.canvas.manager.set_window_title(fig_label)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.show()

def EXI_cal(B,G,R):
    g = G/(R+G+B)
    b = B/(R+G+B)
    r = R/(R+G+B)
    EXG = 2*g-r-b
    EXR = 1.4*r-g
    VARI = EXG-EXR
    return EXG,EXR,VARI
