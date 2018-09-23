import matplotlib.pyplot as plt
import numpy as np


def plot_satellite_image(band_data, plant_data,
                         tree_loc=None, plot_all=True,
                         colorbands=[0, 1, 2]):
    '''
    Description:
        Plots the raster bands, the plant data, and tree locations
    Inputs:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
        plant_data (np.array, size=[r_del, c_del], optional): The rastered
            image data for all bands
        tree_loc (np.array, size=[n, 2], optional): tree locations. If none,
            don't plot
        colorbands (list of ints, optional): color bands to include in color
            image. Dafault [0, 1, 2] == [r, g, b]
        plot_all (bool, optional): plot all colors if true. If false,
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # set up axis depending on flags
    if plot_all is True:
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = np.reshape(axs, [6, ])
        color_scheme = ['red', 'green', 'blue',  'IR']
        # loop over bands
        for index in np.arange(4):
            # plot it
            axs[index].imshow(band_data[:, :, index])
            axs[index].set_title(color_scheme[index])
        plot_ind = 4
    else:
        fig, axs = plt.subplots(1, 2, figsize=(18, 12))
        axs = np.reshape(axs, [2, ])
        plot_ind = 0
    # plot  all colors
    axs[plot_ind].imshow(band_data[:, :, colorbands])
    axs[plot_ind].set_title('All colors')
    plot_ind += 1
    axs[plot_ind].imshow(plant_data)
    axs[plot_ind].set_title('All colors')
    # plot trees
    if tree_loc is not None:
        # plot trees (x and y are switched)
        axs[plot_ind].scatter(tree_loc[:, 1], tree_loc[:, 0], color='r')
        axs[plot_ind].set_title('Plant detect')
