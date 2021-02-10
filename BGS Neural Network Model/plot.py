import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *


class Plot:

    def __init__(self, output_directory_main):

        self.output_directory_main = output_directory_main
        if not os.path.exists(self.output_directory_main):
            os.mkdir(output_directory_main)

    def plot_single_image(self,
                          array,
                          title="",
                          show_plot=True,
                          save_plot=False,
                          sub_folder="",
                          mode="default",
                          name="default",
                          suffix="default"):
        """ plot a numpy array """
        plt.imshow(array)
        plt.title(title)

        if show_plot:
            plt.show()

        if save_plot and save_plot_name is not None:
            self.plot_save(name=save_plot_name, sub_folder=save_plot_folder, )

    def plot_batch(self, batch_array, save=False):
        """ plots a batch if images """
        figure, subplots_axes = self.get_axes(n_rows=1, n_cols=batch_array.shape[0])
        batch_array = np.expand_dims(batch_array, 0)
        for index, subplots_ax in np.ndenumerate(subplots_axes):
            r, c = index
            subplots_ax.imshow(batch_array[r, c])
        plt.show()
        if save:
            self.plot_save(plt)

    @staticmethod
    def get_axes(n_rows, n_cols):
        """ returns an axes array of shape n_rows, n_cols """
        figure, sub_axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4, n_rows * 4))

        if n_rows == 1 and n_cols == 1:
            return figure, np.array([[sub_axes]])
        if n_rows == 1 and n_cols > 1:
            return figure, np.expand_dims(sub_axes, 0)
        if n_rows > 1 and n_cols == 1:
            return figure, np.expand_dims(sub_axes, 1)

        return figure, sub_axes

    def plot_groups(self,
                    groups: list,
                    title=None,
                    descriptions=("BG", "Input", "True", "Pred"),
                    sub_folder_name="subFolder"):
        """ plots a group of images"""
        n_rows = len(groups)
        n_cols = groups[0].shape[0]
        figure, subplot_axes = self.get_axes(n_rows, n_cols)

    def get_directory_count(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            return 0
        return len(os.listdir(directory))

    def plot_save(self, sub_folder="", mode="default", name="default", suffix="default"):
        """ save plots to directory
            main_outputs_folder / sub_folder / mode / name + suffix + .png
        """
        path = os.path.join(self.output_directory_main, sub_folder, mode)
        prefix = self.get_directory_count(path)
        path = os.path.join(path, f"{prefix} + {name} + {suffix} + '.png'")
        plt.savefig(path)
        plt.close()
        print(f"figure saved to {path}")


if __name__ == "__main__":
    plot = Plot(output_directory_main="outputs/")
    sample_image_file = "sample_inputs/bg_boats.jpg"


    def unit_test1():
        for r, c in [(1, 1), (1, 5), (5, 1), (5, 5)]:
            fig, axes = plot.get_axes(r, c)
            assert axes.shape == (r, c), "get axes shape mismatch"


    def unit_test2():
        # unit test for single_image_test
        sample_image = read_image(sample_image_file)
        plot.plot_single_image(array=np.uint8(sample_image), title="sample_image3", save_plot=True,
                               save_plot_name="sample_image2",
                               show_plot=False, )


    unit_test2()
