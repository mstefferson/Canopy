import matplotlib.pyplot as plt
import numpy as np


def plot_satellite_image(band_data, plant_data, tree_loc):
    '''
    Description:
        Plots the raster bands, the plant data, and tree locations
    Inputs:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
        plant_data (np.array, size=[r_del, c_del]): The rastered image
            data for all bands
    Returns:
        plant_data (np.array, size=[r_del, c_del]): A map that is strongly
            peaks where there are trees
        plant_data (np.array, size=[r_del, c_del]): A map that is strongly
            peaks where there are trees
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # st.title('Sample satellite images')
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs = np.reshape(axs, [6, ])
    color_scheme = ['red', 'green', 'blue',  'IR']
    for index in np.arange(4):
        # plot it
        plt.subplot(2, 3, index+1)
        plt.title(color_scheme[index])
        imgplot = plt.imshow(band_data[:, :, index])
    # plot it
    plt.subplot(2, 3, 5)
    plt.title('All colors')
    imgplot = plt.imshow(band_data[:, :, :3])
    # plot it
    plt.subplot(2, 3, 6)
    imgplot = plt.imshow(plant_data)
    # plot trees (x and y are switched)
    plt.scatter(tree_loc[:, 1], tree_loc[:, 0], color='r')
    plt.title('Plant detect')
    plt.show()
    # st.pyplot()
