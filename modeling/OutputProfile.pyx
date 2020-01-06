import matplotlib.pyplot as plt
import numpy as np
import os


class ChannelOutputProfile:
    def __init__(self, directory="", keep_data=False, show_plots=False, save_plots=False):
        self.directory = directory
        self.keep_data = keep_data or save_plots
        self.save_plots = save_plots


class CompartmentOutputProfile:
    def __init__(self, directory="", keep_data=False, show_plots=False, save_plots=False):
        self.directory = directory
        self.keep_data = keep_data or save_plots
        self.save_plots = save_plots


def plot_data(directory, title, data_label, data, delta_t, save=False, show=False):
    time = np.arange(0, len(data) * delta_t, delta_t)
    plt.plot(time, data)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel(data_label)
    plt.minorticks_on()
    plt.savefig(os.path.join(directory, title))
    plt.close()


def plot_axes(directory, title, x_label, y_label, x_data, y_data):
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.minorticks_on()
    plt.savefig(os.path.join(directory, title))
    plt.close()