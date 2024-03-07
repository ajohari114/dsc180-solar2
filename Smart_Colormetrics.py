import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from py2neo import Graph, Node
from scipy.ndimage import gaussian_filter


website = 'fenninggroupnas.ucsd.edu'
port = 7687

graph = Graph(f"bolt://{website}:{port}", auth=("neo4j", "magenta-traffic-powder-anatomy-basket-8461")) # magenta-etc is the passphrase


def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        # black
        return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE
# test = df['img'][6][300,:,:,:]/255

def rgb_to_cmyk(rgb):
	r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
	k = 1 - np.max(rgb, axis=-1)
	c = (1-r-k)/(1-k)
	m = (1-g-k)/(1-k)
	y = (1-b-k)/(1-k)
	return np.dstack([c, m, y, k])


def rgb_to_cmyk(rgb):
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    k = 1 - np.max(rgb, axis=-1)
    c = (1-r-k)/(1-k)
    m = (1-g-k)/(1-k)
    y = (1-b-k)/(1-k)
    return np.dstack([c, m, y, k])


def calculate_K_mean(vm):
    K_means = []
    for frame in range(vm.shape[0]):
        # Convert the frame to CMYK
        cmyk_image = rgb_to_cmyk(vm[frame, :, :, :]/255)
        
        # Extract the K channel
        K_channel = cmyk_image[:,:,3]
        
        # Calculate the 2.5th and 97.5th percentiles (95% CI)
        ci_low, ci_high = np.percentile(K_channel, [2.5, 97.5])
        
        # Mask the values outside the CI
        masked_K_channel = np.ma.masked_outside(K_channel, ci_low, ci_high)
        
        # Calculate the mean of the values within the CI
        K_mean = np.ma.mean(masked_K_channel)
        
        K_means.append(K_mean)
        
    return K_means


def apply_perspective_transform(frame, M, cols, rows):
    return cv2.warpPerspective(frame, M, (cols, rows))


def get_image_creation_date(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'DateTimeOriginal':
                        return value
    except Exception as e:
        print(f"Error reading EXIF data for {image_path}: {e}")
    return None

def crop_image(img, crop_params):
    x0, x1 = crop_params[0]
    y0, y1 = crop_params[1]
    return img[x0:x1, y0:y1]

def create_video_matrix(slice_dict, fids):
    
    # first_image = cv2.imread(fids_0[folder_name][0][0])
    # height, width, channels = first_image.shape

    # # Initialize the video matrix
    num_frames = len(fids)
    video_matrix = None
    
    if 'latest_template.jpg' in os.listdir():
        prev_template = cv2.imread('latest_template.jpg', 0)
    else:
        prev_template = cv2.imread('baseline_template.jpg', 0)
        
    w, h = prev_template.shape[::-1]
    
    # Read each image, crop it, and populate the video matrix
    for i, info in enumerate(tqdm(fids,desc = f'Creating Video Matrix', leave = False)):
        
        image_path = info[0]
            
        frame = cv2.imread(image_path, 0)

        result = cv2.matchTemplate(frame, prev_template, cv2.TM_SQDIFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        bottom_right = (min_loc[0] + w, min_loc[1] + h)

        crop_params = ((min_loc[1],min_loc[1]+h),(min_loc[0],min_loc[0]+w))

        prev_template = crop_image(frame, crop_params)
  
        frame = cv2.imread(image_path)

        frame = crop_image(frame, crop_params)
        
        height, width, channels = frame.shape
        if video_matrix is None:
            video_matrix = np.zeros((num_frames, height, width, channels), dtype=np.uint8)

        video_matrix[i] = frame
        
        if i + 1 == num_frames:
            cv2.imwrite('latest_template.jpg', prev_template)

    return video_matrix

def transform_vm(video_matrix):
    # Define the points in the original image (corners of the original image)
    transformed = np.zeros_like(video_matrix)
    rows, cols = video_matrix.shape[1:3]
    pts1 = np.float32([[cols//31, rows//8.75], [cols-cols//20, rows//35], [0, rows], [cols, rows-rows//4]])  # top left, top right, bottom left, bottom right
    #pts1 = np.float32([[20, 15], [cols-30, 5], [0, rows], [cols, rows-40]])  
    # Define the magnitude of the transformation (e.g., 0.05 for a 5% stretch at the top)
    magnitude = 0.01

    # Define where those points will be in the transformed image (stretching the top)
    pts2 = np.float32([[0, -rows * magnitude], [cols - 1, -rows * magnitude], [0, rows - 1], [cols - 1, rows - 1]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # If the transformed image looks good, uncomment the following lines to apply the transformation to the whole video_matrix
    for i in range(video_matrix.shape[0]):
        transformed[i, :, :, :] = apply_perspective_transform(video_matrix[i, :, :, :], M, cols, rows)
    
    return transformed

def process_pictures(image_file_path, b_id='None'):
    creation_time = []
    image_folder_0 = image_file_path
    go_pro_folders = os.listdir(image_folder_0)[::-1]
    go_pro_folders = np.sort(np.array(go_pro_folders))
    fids_0 = {}
    
    for folder in go_pro_folders:
        if folder not in fids_0:
            fids_0[folder] = []
        folder_path = os.path.join(image_folder_0, folder)
        images = np.array(os.listdir(folder_path))
        images = np.sort(images)
        for image in images:
            if 'control' not in image and '_rgb' not in image and 'ipynb' not in image:
                image_path = os.path.join(folder_path, image)
                creation_time = get_image_creation_date(image_path)
                fids_0[folder].append((image_path, creation_time))
                
    all_frames = []
                
    for i in fids_0.keys():
        all_frames.extend(fids_0[i])
            
    if f'{b_id}_cmet.csv' in os.listdir():
        curr_cm = pd.read_csv(f'{b_id}_cmet.csv', index_col=0)
                
        prev_n_frames = len(curr_cm)
        
        if prev_n_frames == len(all_frames):
            return pd.read_csv(f'{b_id}_cmet.csv', index_col=0)
        elif prev_n_frames < len(all_frames):
            all_frames = all_frames[prev_n_frames:]
        
    slice_dict = {}
    
    center_points_0 = np.full((4, 10), None, dtype=object)
        
    #rows, cols = temp_matrix.shape[1:3]
    
    rows = 185
    
    cols = 620
    
    for i, y in enumerate(np.linspace(0+15,rows-rows//12,4)):
        for j, x in enumerate(np.linspace(0 + 30, cols-cols//21, 10)):
            center_points_0[i,j] = (int(x),int(y))
    
    sample_names_0 = np.arange(0,40).reshape(4,10)
    
    window_size = 25
    samples_to_ignore = [17,22,38,39]
    used_sample_names = []
    for s in sample_names_0.flatten():
        if s not in samples_to_ignore:
            used_sample_names.append(s)

    c_data = pd.DataFrame(columns=used_sample_names)

    # Create a slice matrix to hold the slices
    slice_shape = (2 * window_size, 2 * window_size, 3)
    slice_matrix = np.empty((*sample_names_0.shape, *slice_shape))

    K_means = defaultdict(list)

    K_means['Hour'] = []


    s_time = datetime.datetime.strptime(fids_0[go_pro_folders[0]][0][1], '%Y:%m:%d %H:%M:%S')
    
    # Extract the slices from the video_matrix using the center points
    
    
    video_matrix = create_video_matrix(slice_dict, all_frames)
    video_matrix = transform_vm(video_matrix)
    num_frames = video_matrix.shape[0]
    for p in tqdm(range(num_frames), desc = f'Calculating Colormetrics', leave = False):
        picture_time = datetime.datetime.strptime(all_frames[p][1], '%Y:%m:%d %H:%M:%S')
        K_means['Hour'].append(np.round(((picture_time-s_time).total_seconds() / 60**2),3))
        for i in range(sample_names_0.shape[0]):
            for j in range(sample_names_0.shape[1]):
                if sample_names_0[i][j] in samples_to_ignore:
                    continue
                x, y = center_points_0[i, j]

                # Boundary checks
                x_min = max(0, x - window_size)
                x_max = min(video_matrix.shape[2], x + window_size)
                y_min = max(0, y - window_size)
                y_max = min(video_matrix.shape[1], y + window_size)

                temp_slice = rgb_to_cmyk(video_matrix[p, y_min:y_max, x_min:x_max, :])/255
                # print(temp_slice)
                K_channel = temp_slice[:,:,3]
                # Calculate the 2.5th and 97.5th percentiles (95% CI)
                ci_low, ci_high = np.percentile(K_channel, [2.5, 97.5])

                # Mask the values outside the CI
                masked_K_channel = np.ma.masked_outside(K_channel, ci_low, ci_high)
                # Calculate the mean of the values within the CI
                K_mean = np.ma.mean(masked_K_channel)


                # Padding in case the slice is smaller than the window
                pad_x_min = window_size - (x - x_min)
                pad_x_max = window_size + (x_max - x)
                pad_y_min = window_size - (y - y_min)
                pad_y_max = window_size + (y_max - y)


                K_means[f'sample{sample_names_0[i][j]}'].append(1-K_mean)
                # slice_matrix[i, j, pad_y_min:pad_y_max, pad_x_min:pad_x_max, :] = temp_slice
    return pd.DataFrame(K_means)

def generate_csv(fp, b_id):
    df = process_pictures(fp,b_id)
    if f'{b_id}_cmet.csv' in os.listdir():
        prev_df = pd.read_csv('b40_cmet.csv', index_col = 0)

        new_csv = pd.concat([prev_df,df]).reset_index(drop = True)
    
        new_csv.to_csv(f'{b_id}_cmet.csv', index = False)
    else:
        df.to_csv(f'{b_id}_cmet.csv', index = False)
        
def get_curve_params(x,y):
    def logistic(x, L=1, x_0=0, k=1):
        return L / (1 + np.exp(-k * (x - x_0)))
    L_estimate = 0.6
    x_0_estimate = 200
    k_estimate = .03
    p_0 = [L_estimate, x_0_estimate, k_estimate]
    
    smoothed = gaussian_filter(y, sigma = 1000)


    popt, _ = curve_fit(logistic, x, smoothed, p_0, bounds = ([0, -np.inf, -np.inf],[1, np.inf, np.inf]))
    pred_cmet = logistic(range(800), *popt)
    pred_cmet = (pred_cmet-np.min(pred_cmet))/(popt[0]-np.min(pred_cmet))
    
    og_L = popt[0]
        
    p_0[2] = popt[2]
    p_0[1] = popt[1]
    p_0[0] = 1
    
    popt, _ = curve_fit(logistic, range(800), pred_cmet, p_0, bounds = ([1, popt[1]-1, 0],[1.0001, popt[1], popt[2]]))
    
    return popt, float(og_L)

def create_cmet_node(batch_id, sample_id, cmet, og_L, curve_L, curve_x0, curve_k):
    graph.run(f"""MATCH (n)
                  WHERE n.action = 'colormetrics' and n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}'
                  DELETE n""")
    new_node = Node('Action', action = 'colormetrics', batch_id = batch_id, sample_id = sample_id, colormetrics_hours = [i[0] for i in cmet], normalized_colormetrics = [i[1] for i in cmet], curve_L = curve_L, curve_x0= curve_x0, curve_k = curve_k, original_L = og_L)
    
    graph.create(new_node)
    
def add_cmet_data(fp, b_id):
    cmet = pd.read_csv(fp)
    
    if 'Unnamed' in cmet.columns[0]:
        cmet = cmet.iloc[:,1:]

    for i in tqdm(cmet.columns[1:]):
        if i == 'Hour':
            continue
        cmet[i] = (cmet[i] - np.min(cmet[i]))
        curve_params, og_L = get_curve_params(cmet['Hour'], cmet[i])
        curve_params = [float(i) for i in curve_params]

        curr_cmet = cmet[['Hour', i]].values.tolist()
        curr_cmet = [[float(j) for j in i] for i in curr_cmet]
        create_cmet_node(b_id, i, curr_cmet, og_L, *curve_params)
