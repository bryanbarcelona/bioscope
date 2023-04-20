import os, shutil
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import numpy as np
from PIL import Image
from skimage import feature, segmentation, exposure
from cellpose import core, utils, io, models, metrics

module_dir = os.path.dirname(__file__)

model_rel_path = "models\\CP_20230410_001152"

model_path = os.path.join(module_dir, model_rel_path)


def detect_cells(folder_path, model_path=model_path):

    filenames = glob.glob(f'{folder_path}\\*.jpg')

    # Create the directory if it does not exist
    if not os.path.exists(os.path.join(folder_path, 'detected')):
        os.makedirs(os.path.join(folder_path, 'detected'))

    for filename in filenames:

        filename_raw = filename.split("\\")[-1].split(".")[0]
        image = Image.open(filename)
        img_array = np.array(image)

        model = models.CellposeModel(gpu=True, 
                                    pretrained_model=model_path)

        diameter = 300
        flow_threshold = 0.4
        cellprob_threshold= 0
        chan = 1
        chan2 = 2

        # use model diameter if user diameter is 0
        diameter = model.diam_labels if diameter==0 else diameter

        # run model on test images
        masks, flows, styles = model.eval(img_array, 
                                        channels=[chan, chan2],
                                        diameter=diameter,
                                        flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold
                                        )

        io.save_masks(img_array, masks, flows, file_names=filename, 
            channels=[chan, chan2],
            png=True, # save masks as PNGs and save example image
            tif=True, # save masks as TIFFs
            save_txt=True, # save txt outlines for ImageJ
            save_flows=True, # save flows as TIFFs
            save_outlines=True, # save outlines as TIFFs 
            )
        
        # Create a 1x3 grid of subplots
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        print("Flows:", type(flows[4]), len(flows))
        print("Flows shapes:", flows[0].shape, flows[1].shape, flows[2].shape, flows[3].shape)
        print("Flows[0]:", type(flows[0]), flows[0].shape, flows[0].dtype)
        print("Flows[0][0]:", type(flows[0][0]), flows[0][0].shape)
        print("Flows[0][1]:", type(flows[0][1]), flows[0][1].shape)
        print("Flows[0][2]:", type(flows[0][2]), flows[0][2].shape)
        print("Flows[0][3]:", type(flows[0][3]), flows[0][3].shape)
        #np.savetxt('D:\\Work\\CellDetection2\\flows\\Flows0.txt', flows[0])
        #np.savetxt('D:\\Work\\CellDetection2\\flows\\Flows00.txt', flows[0][0])
        #np.savetxt('D:\\Work\\CellDetection2\\flows\\Flows01.txt', flows[0][1])
        #np.savetxt('D:\\Work\\CellDetection2\\flows\\Flows02.txt', flows[0][2])
        #np.savetxt('D:\\Work\\CellDetection2\\flows\\Flows03.txt', flows[0][3])
        io.masks_flows_to_seg(img_array, masks, flows, diams=300, file_names=filename)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.imshow(flows[3][0], cmap='gray')
        ax2.imshow(flows[3][1], cmap='jet')
        plt.show()
        plt.imshow(np.uint8(flows[0]))
        plt.show()

        hsv_image = flows[0]

        # Convert to RGB color space for plotting
        rgb_image = colors.hsv_to_rgb(hsv_image)

        # Get a mask for black pixels
        black_mask = np.all(hsv_image == [0, 0, 0], axis=-1)

        # Set black pixels to white (or any other color) in the RGB image
        rgb_image[black_mask] = [1, 1, 1]

        # Plot the image
        plt.imshow(rgb_image)
        plt.show()

        mask = np.ones_like(flows[0], dtype=np.uint8) * 255  # create a new numpy array with same shape as flows[0] and fill with 255
        mask[flows[0] == 0] = 0  # set black pixels to 0

        print("mask:", type(mask), mask.shape, mask.dtype)
        # create new array with black and white colors
        #new_arr = np.zeros_like(flows[0])
        #new_arr[mask] = 255

        # plot the image with black and white colors
        plt.imshow(mask, cmap='gray')
        plt.show()

        # Plot the images in each subplot and save as PNG
        axs[0].imshow(img_array, cmap='gray')
        axs[0].set_title('Image')
        axs[1].imshow(masks, cmap='gray')
        axs[1].set_title('Masks')
        axs[2].imshow(flows[0], cmap='gray')
        axs[2].set_title('Flows')
        plt.savefig(f'{folder_path}\\detected\\{filename_raw}.png', dpi=300, bbox_inches='tight')
        print("done")
    # clean up - deleting pre process
    if os.path.exists(os.path.join(folder_path, 'processed')):
        shutil.rmtree(f'{folder_path}\\processed')