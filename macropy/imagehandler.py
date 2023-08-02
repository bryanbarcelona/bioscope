import os
import shutil
import glob
from readlif.reader import LifFile
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import tifffile
import oiffile
import tqdm

def tiffify_stk(nd_path):

    def image_tzcs(nd):

        # Initializing the nd dictionary and filling it with the information provided in the nd text file
        nd_dict = {}
        with open(nd) as nd:
            for line in nd:
                if ',' in line:
                    key, value = line.strip().split(',', maxsplit=1)
                    key = key.strip('"')
                    value = value.strip()
                    nd_dict[key] = value
        
        # Adding dummy values when dictionary entry doesn not exist
        if 'ZStepSize' not in nd_dict:
            nd_dict['ZStepSize'] = '0'
        if 'NZSteps' not in nd_dict:
            nd_dict['NZSteps'] = '1'

        if nd_dict['DoStage'] == 'FALSE':
            nstage = 1
            stage_status = 'FALSE'
        elif nd_dict['DoStage'] == 'TRUE':
            nstage = int(nd_dict['NStagePositions'])
            stage_status = 'TRUE'
        if nd_dict['DoWave'] == 'FALSE':
            nwave = 1
            wave_status = 'FALSE'
        elif nd_dict['DoWave'] == 'TRUE':
            nwave = int(nd_dict['NWavelengths'])
            wave_status = 'TRUE'
        if nd_dict['DoZSeries'] == 'FALSE':
            nzseries = 1
            z_status = 'FALSE'
            stepsize = 1
        elif nd_dict['DoZSeries'] == 'TRUE' and float(nd_dict['ZStepSize'].replace(",", ".")) == 0:
            nzseries = 1
            z_status = 'FALSE'
            stepsize = 1
        elif nd_dict['DoZSeries'] == 'TRUE' and float(nd_dict['ZStepSize'].replace(",", ".")) > 0:
            nzseries = int(nd_dict['NZSteps'])
            z_status = 'TRUE'
            stepsize = float(nd_dict['ZStepSize'].replace(",", "."))
        if nd_dict['DoTimelapse'] == 'FALSE'and float(nd_dict['ZStepSize'].replace(",", ".")) > 0:
            ntimepoints = 1
            time_status = 'FALSE'
        elif nd_dict['DoTimelapse'] == 'FALSE' and float(nd_dict['ZStepSize'].replace(",", ".")) == 0:
            ntimepoints = int(nd_dict['NZSteps'])
            time_status = 'TRUE'
            for z in range(nzseries):
                nd_dict[f'WaveDoZ{z+1}'] = 'FALSE'
        elif nd_dict['DoTimelapse'] == 'TRUE':
            ntimepoints = int(nd_dict['NTimePoints'])
            time_status = 'TRUE'
        
        # Reassigning the corrected metadata to the nd dictionary
        nd_dict['NStagePositions'] = nstage
        nd_dict['NWavelengths'], nd_dict[' SizeC'] = nwave, nwave
        nd_dict['NZSteps'], nd_dict[' SizeZ'] = nzseries, nzseries
        nd_dict['NTimePoints'], nd_dict[' SizeT'] = ntimepoints, ntimepoints
        nd_dict['DoStage'] = stage_status
        nd_dict['DoWave'] = wave_status
        nd_dict['DoZSeries'] = z_status
        nd_dict['DoTimelapse'] = time_status
        nd_dict['ZStepSize'] = stepsize

        TZCS = (ntimepoints, nzseries, nwave, nstage)
        # Configuration outputs a tuple Spacing - nChannel - nFrames - nSlices for the tifffile.imwrite metadata
        config = (float(nd_dict['ZStepSize']), nwave, ntimepoints, nzseries)
        
        return TZCS, nd_dict, config

    def get_configurations(nd, nd_dict):
        expermiment_path = os.path.splitext(nd)[0]
        image_files = glob.glob(f'{expermiment_path}*.[ts][ti][fk]')

        # Initiating the static info for each image TIFF
        constant_info = {' DimensionOrder': 'XYCZT', ' IsInterleaved': 'false', ' IsRGB': 'false', ' LittleEndian': 'true'}

        # Getting the metadata using the first image file (tif or stk) in the image series
        with tifffile.TiffFile(image_files[0]) as tif:
            ifd_tags = dict(tif.pages[0].tags)
            sub_dicts = {k: {'name': ifd_tags[k].name, 'value': ifd_tags[k].value, 'offset': ifd_tags[k].offset} 
                    for k in ifd_tags}

            pixel_type = {' PixelType': str(tif.pages[0].dtype)}

            if 256 in sub_dicts:
                width = {'ImageWidth': str(sub_dicts[256]['value'])}
                sizex = {' SizeX': str(sub_dicts[256]['value'])}
            if 257 in sub_dicts:
                length = {'ImageLength': str(sub_dicts[257]['value'])}
                sizey = {' SizeY': str(sub_dicts[257]['value'])}           
            if 258 in sub_dicts and 277 in sub_dicts:
                bits_per_sample = sub_dicts[258]['value']
                bits_per_pixel = {' BitsPerPixel': str(int(bits_per_sample/sub_dicts[277]['value']))}
                samples_per_pixel = {'SamplesPerPixel': str(sub_dicts[277]['value'])}
                bits_per_sample = {'BitsPerSample': str(bits_per_sample)}
            if 259 in sub_dicts:
                if sub_dicts[259]['value'] == 1:
                    compression = {'Compression': 'Uncompressed'}
            if 262 in sub_dicts:
                if sub_dicts[262]['value'] == 1:
                    photometric = {'PhotometricInterpretation': 'BlackIsZero'}
            if 282 in sub_dicts:
                x_res = {'XResolution': "{:.1f}".format(float(sub_dicts[282]['value'][0]))}
            if 283 in sub_dicts:
                y_res = {'YResolution': "{:.1f}".format(float(sub_dicts[283]['value'][0]))}
            if 296 in sub_dicts:
                if sub_dicts[296]['value'] == 3:
                    res_unit = {'ResolutionUnit': 'Centimeter'}
            if 305 in sub_dicts:
                software = {'Software': str(sub_dicts[305]['value'])}
            if 33628 in sub_dicts:
                datetime = {'DateTime': str(sub_dicts[33628]['value']['CreateTime'].strftime("%Y:%m:%d %H:%M:%S"))}
                x_cal = {'XCalibration': float(sub_dicts[33628]['value']['XCalibration'])}
                y_cal = {'YCalibration': float(sub_dicts[33628]['value']['YCalibration'])}

            # Recalculating X/Y calibration when these are 0.0
            if x_cal['XCalibration'] == 0:
                x_cal['XCalibration'] = 1/(float(x_res['XResolution'])/10000)
            if y_cal['YCalibration'] == 0:
                y_cal['YCalibration'] = 1/(float(y_res['YResolution'])/10000)

            # Assembling the info dictionary
            info_dictionary = {}
            info_dictionary.update(constant_info)
            info_dictionary.update(pixel_type)
            info_dictionary.update(width)
            info_dictionary.update(sizex)
            info_dictionary.update(length)
            info_dictionary.update(sizey)          
            info_dictionary.update(bits_per_pixel)
            info_dictionary.update(samples_per_pixel)
            info_dictionary.update(bits_per_sample)
            info_dictionary.update(compression)
            info_dictionary.update(photometric)
            info_dictionary.update(x_res)
            info_dictionary.update(y_res)
            info_dictionary.update(x_cal)
            info_dictionary.update(y_cal)
            info_dictionary.update(res_unit)
            info_dictionary.update(software)
            info_dictionary.update(datetime)
            info_dictionary.update(nd_dict)

            # Sorting the info dictionary by key alphabetically
            sorted_keys = sorted(info_dictionary.keys(), key=str.lower)
            info_dictionary = {key: info_dictionary[key] for key in sorted_keys}

            # Turning the info dictionary into a raw string
            info_str = ""
            for key, value in info_dictionary.items():
                info_str += f'{key} = {value}\r\n'
            return info_str, (x_cal['XCalibration'], y_cal['YCalibration'])

    def image_assembly(nd, tzcs, info, xy_cal, config):

        # Generating a save method for the arrays assembled through various routes
        def save_tif(image, path, info, xy_cal, config, ranges, min_range, max_range):
            tifffile.imwrite(path, 
                            image,
                            photometric='minisblack', 
                            imagej=True,  # Add ImageJ metadata
                            resolution=(1./xy_cal[0], 1./xy_cal[1]),
                            metadata={
                                    "axes": "TZCYX",  # Specify the order of the axes
                                    "spacing": config[0],  # Specify the pixel spacing
                                    "unit": "micron",  # Specify the units of the pixel spacing
                                    "hyperstack": "true",  # Specify that the data is a hyperstack
                                    "mode": "color",  # Specify the display mode
                                    "channels": config[1],  # Specify the channel colors
                                    "frames": config[2],  # Specify the number of frames
                                    "Info": info,
                                    "slices": config[3],  # Specify the number of slices
                                    "Ranges": ranges,
                                    'min': min_range, 
                                    'max': max_range,
                                    "metadata": "ImageJ=1.53c\n",  # Add a blank metadata field for ImageJ
                                })
        
        # Getting all the image files associated with the .nd file
        expermiment_path = os.path.splitext(nd)[0]
        #image_files = glob.glob(f'{expermiment_path}*.[ts][ti][fk]')
        tiff_path = os.path.splitext(nd)[0] + ".tif"

        # Loading the TZCS configuration passed from image_tzcs()
        TZCS = tzcs

        # Iterating through the number of stages acquired
        for series in tqdm.trange(TZCS[3]):
            
            # Gathering the image files relevant for each stage position
            if TZCS[3] == 1:
                series_image_files = glob.glob(f'{expermiment_path}*.[ts][ti][fk]')
            elif TZCS[3] > 1:
                series_image_files = glob.glob(f'{expermiment_path}*_s{series+1}_*.[ts][ti][fk]')
                tiff_path = f"{os.path.splitext(nd)[0]} Series {series+1}.tif"
            print(f'Series {series+1}:')

            # Initializing the image_array which is later passed to tifffile.imwrite
            image_array = []

            # Determining if the time dimension was saved as .stk or individual .tif and assigning a case value (1-3)
            for timepoint in series_image_files:
                if '_t' not in timepoint and TZCS[0] > 1 and TZCS[1] == 1 and len(series_image_files) == TZCS[2] and '.stk' in timepoint: 
                    time_as_stk = 1
                elif '_t' not in timepoint and TZCS[0] == 1:
                    time_as_stk = 2
                elif '_t' in timepoint:
                    time_as_stk = 3
            
            time_image_series = []
            complete_image = []
            if time_as_stk == 1:
                for timepoint in series_image_files:
                    stk_file = tifffile.imread(timepoint)
                    time_image_series.append(stk_file)
                time_image_series = np.stack(time_image_series, axis=0)
                complete_image = np.transpose(time_image_series, (1, 0, 2, 3))[:, np.newaxis, :, :, :]
                print(f"Case 1 Shape: {time_image_series.shape}")
                
                ranges = [[],[]]
                for c in range(complete_image.shape[2]):
                    channel_data = complete_image[:, :, c, :, :]
                    channel_min = np.amin(channel_data)
                    channel_max = np.amax(channel_data)
                    ranges[0].append(np.amin(channel_data))
                    ranges[1].append(np.amax(channel_data))

                min_value = float("{:.1f}".format(min(ranges[0])))
                max_value = float("{:.1f}".format(min(ranges[1])))

                # Creating the range tuple
                range_tuple = ()
                for r in range(len(ranges[0])):
                    range_tuple = range_tuple + (float("{:.1f}".format(ranges[0][r])),)
                    range_tuple = range_tuple + (float("{:.1f}".format(ranges[1][r])),)

                save_tif(complete_image, tiff_path, info, xy_cal, config, range_tuple, min_value, max_value)

            elif time_as_stk == 2:
                #time_image_series.append(series_image_files)
                for channel in series_image_files:
                    stk_file = tifffile.imread(channel)
                    complete_image.append(stk_file)
                complete_image = np.stack(complete_image, axis=0).astype(np.uint16)
                if len(complete_image.shape) == 3:
                    complete_image = np.expand_dims(complete_image, axis=1)
                print(complete_image.shape)
                complete_image = np.transpose(complete_image, (1, 0, 2, 3))[np.newaxis, :, :, :, :]
                print(complete_image.shape)
                #print(len(time_image_series), len(time_image_series[0]))
                print("Case 2")

                ranges = [[],[]]
                for c in range(complete_image.shape[2]):
                    channel_data = complete_image[:, :, c, :, :]
                    channel_min = np.amin(channel_data)
                    channel_max = np.amax(channel_data)
                    ranges[0].append(np.amin(channel_data))
                    ranges[1].append(np.amax(channel_data))
                    #print(f"{tiff_file} Channel {c}: min={channel_min}, max={channel_max}")
                #print(ranges)

                min_value = float("{:.1f}".format(min(ranges[0])))
                max_value = float("{:.1f}".format(min(ranges[1])))

                # Creating the range tuple
                range_tuple = ()
                for r in range(len(ranges[0])):
                    range_tuple = range_tuple + (float("{:.1f}".format(ranges[0][r])),)
                    range_tuple = range_tuple + (float("{:.1f}".format(ranges[1][r])),)

                save_tif(complete_image, tiff_path, info, xy_cal, config, range_tuple, min_value, max_value)

            elif time_as_stk == 3:
                for t in tqdm.trange(TZCS[0]):
                    time_notation_dot = f'_t{t+1}.'
                    time_notation_underscore = f'_t{t+1}_'
                    # Find and add file paths that match the pattern
                    timepoint_files = [timepoint for timepoint in series_image_files if time_notation_dot in timepoint or time_notation_underscore in timepoint]
                    time_image_series.append(timepoint_files)             
                
                for t in range(len(time_image_series)):
                    # Initializing a channel list in which the stacks (or tif when z = 1) are added for each z plane
                    
                    first_image = tifffile.imread(time_image_series[0][0])
                    image_shape = first_image.shape
                    dummy_array = np.zeros(image_shape, dtype=np.uint16)

                    timepoint_series = []
                    for c in range(TZCS[2]):
                        
                        channel_series = []

                        channel_notation = f'_w{c+1}'
                        channel_notation_dot = f'_w{c+1}.'
                        channel_notation_underscore = f'_w{c+1}_'

                        for channel in time_image_series[t]:
                            if channel_notation in channel or channel_notation_dot in channel or channel_notation_underscore in channel:
                                stk_file = tifffile.imread(channel)
                                channel_series.append(stk_file)
                            elif '_w' not in channel:
                                stk_file = tifffile.imread(time_image_series[t])
                                channel_series.append(stk_file)

                        # If no matching files found, add all files to the list
                        if not channel_series:
                            channel_series.append(dummy_array)
                        
                        channel_series = np.stack(channel_series, axis=0)
                        channel_series = np.squeeze(channel_series, axis=0)
                        print(f"Z Shape (TIME {t} {channel_series.shape} {channel_series.dtype}")
                        timepoint_series.append(channel_series)
                    timepoint_series = np.stack(timepoint_series, axis=0)
                    print(f"Channel Shape {timepoint_series.shape} {timepoint_series.dtype}")
                    complete_image.append(timepoint_series)
                complete_image = np.stack(complete_image, axis=0).astype(np.uint16)
                complete_image = np.expand_dims(complete_image, axis=2)
                complete_image = np.transpose(complete_image, (0, 2, 1, 3, 4))
                print(f"Time Shape {complete_image.shape}")
                print(f"Case 3 Shape: {complete_image.shape}")

                ranges = [[],[]]
                for c in range(complete_image.shape[2]):
                    channel_data = complete_image[:, :, c, :, :]
                    channel_min = np.amin(channel_data)
                    channel_max = np.amax(channel_data)
                    ranges[0].append(np.amin(channel_data))
                    ranges[1].append(np.amax(channel_data))

                min_value = float("{:.1f}".format(min(ranges[0])))
                max_value = float("{:.1f}".format(min(ranges[1])))

                # Creating the range tuple
                range_tuple = ()
                for r in range(len(ranges[0])):
                    range_tuple = range_tuple + (float("{:.1f}".format(ranges[0][r])),)
                    range_tuple = range_tuple + (float("{:.1f}".format(ranges[1][r])),)

                save_tif(complete_image, tiff_path, info, xy_cal, config, range_tuple, min_value, max_value)
    
    nd_path = glob.glob(f'{nd_path}\\*.nd')
    print(f'Found {len(nd_path)} imaging units based on the number of .nd files')
    for nd in nd_path:
        print(os.path.splitext(os.path.basename(nd))[0])
        tzcs, nd_dict, config = image_tzcs(nd)
        info, x_y_config = get_configurations(nd, nd_dict)
        image_assembly(nd, tzcs, info, x_y_config, config)
        
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
