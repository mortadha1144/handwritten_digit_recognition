import matplotlib.pyplot as plt
import numpy as np


def plot_images(X_train, y_train):
    m = X_train.shape[0]
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

    # fig.tight_layout(pad=0.5)
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X_train[random_index]

        # Display the image
        ax.imshow(X_random_reshaped, cmap="gray")

        # Display the label above the image
        ax.set_title(y_train[random_index])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)


def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
