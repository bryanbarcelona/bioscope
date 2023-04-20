import os
import glob
from readlif.reader import LifFile
from PIL import Image, ImageDraw, ImageChops

def preprocess_for_cell_detection(folder_path):
    filenames = glob.glob(f'{folder_path}\\*.lif')

    # Create the directory if it does not exist
    if not os.path.exists(os.path.join(folder_path, 'processed')):
        os.makedirs(os.path.join(folder_path, 'processed'))

    for filename in filenames:

        lif = LifFile(filename)
        image_cells = []
        series_count = len(list(lif.get_iter_image()))
        filename_raw = filename.split("\\")[-1].split(".")[0]

        #print(f'{filename} has {series_count} images.')
        for i in range(series_count):

            lifs = lif.get_image(img_n=i)
            
            z_count = lifs.info["dims_n"][3]
            x_dim = lifs.info["dims_n"][1]
            y_dim = lifs.info["dims_n"][2]
            c_count = lifs.info["channels"]
            
            screened_channels = []
            for c in range(c_count):
                blended_image = Image.new("L", (x_dim, y_dim), 0)
                for z in range(z_count):
                    image = lifs.get_frame(z=z, c=c)

                    # Blend the two images with "screen" blend mode
                    blended_image = ImageChops.screen(blended_image, image)
                    blended_image.save(f'{folder_path}\\processed\\{filename_raw} Series {i+1} channel {c} z {z}.jpg')
                screened_channels.append(blended_image)
                
            # Blend all images in the list with "screen" mode
            blended_image = None
            for image in screened_channels:
                if blended_image is None:
                    blended_image = image
                else:
                    blended_image = ImageChops.multiply(blended_image, image)
            blended_image.save(f'{folder_path}\\processed\\{filename_raw} Series {i+1}.jpg')