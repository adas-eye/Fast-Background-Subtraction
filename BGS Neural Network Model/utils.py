import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import torch
import cv2


def show(matrices,
         title=None,
         descriptions=None,
         root_out_path=None,
         subfolderName="subfolder",
         epochNum=None,
         prefix="",
         show_plots=True
         ):
    """

    matrices : list of batched numpy array
    title : title of plot
    descriptions : plot cols descriptions.
    matrices can contain any number of batched arrays, but in the order of
    bg, raw, true, pred or bg, raw, mask(optional) so that plot labels are correct.
    """
    # -------------
    # draw figure
    # -------------
    n_cols = matrices[0].shape[0]
    n_rows = len(matrices)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4, n_rows * 4))

    if title is not None: fig.suptitle(title, x=0, y=0)

    if descriptions is None and len(matrices) == 4: descriptions = ["bg", "raw", "pred", "true"]
    if descriptions is None and len(matrices) <= 3: descriptions = ["bg", "raw", "mask"]

    for row in range(n_rows):
        for col in range(n_cols):

            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1 and n_cols != 1:
                ax = axes[col]
            elif n_rows != 1 and n_cols == 1:
                ax = axes[row]
            else:
                ax = axes[row][col]

            ax.imshow(matrices[row][col], vmax=1)
            ax.set_axis_off()

            if descriptions is not None:
                ax.set_title(descriptions[row])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    # ---------------
    # save figure 
    # ---------------
    if root_out_path is not None:
        # folder
        if not os.path.exists(root_out_path): os.mkdir(root_out_path)

        folder = os.path.join(root_out_path, "Images")
        if not os.path.exists(folder): os.mkdir(folder)

        # subfolder
        folder = os.path.join(folder, f"{subfolderName}")
        if not os.path.exists(folder): os.mkdir(folder)

        # sub-subfolder
        if epochNum is not None:
            folder = os.path.join(folder, f"epoch {epochNum}")
            if not os.path.exists(folder): os.mkdir(folder)

        # posfix
        files = glob(folder + "/*")
        postfix = f"{len(files) + 1}"

        path = os.path.join(folder, f"{str(prefix) + '-' if prefix is not None else ''}{postfix}.png")
        plt.savefig(path)
        print(f"figure saved to {path}")

    if show_plots:
        plt.show()

    plt.close()


def channel_first(img):
    # H(0) x W(1) x C(2) => C(2) x H(0) x W(1)
    img = convertToNumpy(img)
    return img.transpose((2, 0, 1))


def channel_last(img):
    # C(0) x H(1) x W(2)
    # print(type(img))
    img = convertToNumpy(img)
    # print(type(img))
    return img.transpose((1, 2, 0))


def batch_channel_first(batch_img):
    # b[0] x H[1] x W[2] x C[3]
    batch_img = convertToNumpy(batch_img)
    return batch_img.transpose((0, 3, 1, 2))


def batch_channel_last(batch_img):
    # b[0] x C[1] x H[2] x W[3]
    batch_img = convertToNumpy(batch_img)
    return batch_img.transpose((0, 2, 3, 1))


def convertToNumpy(img):
    if torch.is_tensor(img):
        return img.numpy()
    return img


def read_image(filepath):
    image = np.float32(cv2.imread(filepath))
    try:
        print(image.ndim)
    except:
        print(image)
    return image