import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import re
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import shutil
import time
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')

###########################################################################################
'''
User Defined inputs:
input_path: Folder path containing all the .jp2 files
res_path: Folder path to save all the outputs
jumps: Tunable parameter to define number of jumps during the reordering process
column_data: If the image is expected to have all the text data in columns which can be read top to bottom per column then keep this value as True
model_file_path: Location of the trained YOLOv5 model file
'''
input_path = 'all_jp2'
res_path = 'script_outputs'

jumps = 4
column_data = False

model_file_path = r'trained_model\old_newspaper_model\best_YOLOv5l.pt'

# Command needs to be added to provide the path for the tesseract.exe file
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
###########################################################################################

def process_text_list(txt_list):
    # Some basic text processing on the OCR'd text from images, remaining processing happening in the main loop
    txt_list = [' '.join(i.split('\n')) for i in txt_list]
    txt_list = ' '.join(txt_list)
    return txt_list

def merge_boxes(result_df, threshold):
    result_df = result_df.copy()
    box_tensor = torch.Tensor(result_df.iloc[:, :4].values)
    iou = torchvision.ops.box_iou(box_tensor, box_tensor)
    iou = torch.triu(iou, diagonal = 1)

    # Calculating the area of each box
    box_areas = (result_df['xmax'] - result_df['xmin']) * (result_df['ymax'] - result_df['ymin'])

    # Getting the locations of max. IOU values per row
    vals, idx_1 = torch.max(iou, dim = 1)
    idx_0 = np.arange(13, dtype = 'int')
    idx_1 = np.array(idx_1, dtype = 'int')

    # Iterating over the values to see which values cross the threshold
    for i in list(zip(idx_0, idx_1)):
        score = iou[i[0], i[1]].item()
        if score > threshold:
            # print(i[0], i[1])
            new_xmin = np.min([result_df.loc[i[0], 'xmin'], result_df.loc[i[1], 'xmin']])
            new_ymin = np.min([result_df.loc[i[0], 'ymin'], result_df.loc[i[1], 'ymin']])
            new_xmax = np.max([result_df.loc[i[0], 'xmax'], result_df.loc[i[1], 'xmax']])
            new_ymax = np.max([result_df.loc[i[0], 'ymax'], result_df.loc[i[1], 'ymax']])

            if box_areas[i[0]] > box_areas[i[1]]:
                result_df.loc[i[0], ['xmin', 'ymin', 'xmax', 'ymax']] = new_xmin, new_ymin, new_xmax, new_ymax
                result_df = result_df.drop(i[1])
            else:
                result_df.loc[i[1], ['xmin', 'ymin', 'xmax', 'ymax']] = new_xmin, new_ymin, new_xmax, new_ymax
                result_df = result_df.drop(i[0])

    result_df = result_df.reset_index(drop = True)
    return result_df

def plot_order(image, result_df):
    '''
    Function to visualise the boxes on top of the image and see the current order
    '''
    image = np.array(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in result_df.iterrows():
        vals = i[1][:6].values
        image = cv2.rectangle(image, (vals[0], vals[1]), (vals[2], vals[3]), (0, 0, 255), 10)
        image = cv2.putText(image, str(i[0]), (int(vals[4]), int(vals[5])), font, 15, (0, 0, 255), 15)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def model_inference_full_size(img_path, model_file, txt_model, conf, threshold):
    '''
    What's happening in this function?
    - This function uses the trained YOLOv5 model to detect and generate the boxes for the text and scales them according to the full sized image
    - Uses basic DBSCAN (because I was lazy) to put these boxes into groups
    - Uses Pytesseract to extract text and add that in columns of the dataframe
    - Performs some more text cleaning operations
    - Uses the txt_model loaded in the previous cell to generate sentence embeddings for each of the text

    Outputs:
    - Original full size image
    - Scaled down image
    - Dataframe of boxes, text and embeddings
    '''
    size = 1200

    image = Image.open(img_path)
    w, h = image.size # original image dimensions
    orig_shape = (h, w)

    n_image = image.resize((int(1200/1.367), size))
    n_image = np.array(n_image)
    update_shape = n_image.shape
    
    # Running the model to get the boxes for the image
    res = model_file(n_image, size = 1024)

    # If there is no box detected in the image
    if len(res.pandas().xyxyn[0]) == 0:
        print('No Box detected')
        return None, None, None
    else:
        res = res.pandas().xyxyn[0]
        res.iloc[:, 0] = res.iloc[:, 0] * w
        res.iloc[:, 2] = res.iloc[:, 2] * w
        res.iloc[:, 1] = res.iloc[:, 1] * h
        res.iloc[:, 3] = res.iloc[:, 3] * h
        round_res = res.iloc[:, :4].round(0).astype('int')
        res = pd.concat([round_res, res.iloc[:, 4:]], axis = 1)
        res = res[res['confidence'] >= conf]
        res['x_centers'] = (res.iloc[:, 0] + res.iloc[:, 2])/2
        res['y_centers'] = (res.iloc[:, 1] + res.iloc[:, 3])/2
        res = res.sort_values(['x_centers', 'y_centers']).reset_index(drop = True)

        # Clustering performed to cluster the boxes that share a column together
        clustering = DBSCAN(eps = 160, min_samples = 1, n_jobs = -1).fit(res.iloc[:, -2].values.reshape(-1, 1))
        res['cols'] = clustering.labels_
        
        # Sorting the individual clusters with y_centers column
        temp = pd.DataFrame()
        for i, val in res.groupby('cols'):
            temp = pd.concat([temp, val.sort_values('y_centers')], axis = 0)
        
        res = temp.reset_index(drop = True)
        res = res.drop(['confidence', 'class', 'name'], axis = 1)

        # Using IOU score to merge small and big overlapping boxes
        res = merge_boxes(res, threshold)
        
        # Getting text extracted for each box and adding it in the res dataframe
        image_arr = np.array(image)
        text = []
        for i in res.iterrows():
            vals = i[1].values
            crop = image_arr[int(vals[1]):int(vals[3]), int(vals[0]):int(vals[2])]
            text.append(pytesseract.image_to_string(crop))
        text = np.array(text)
        res['text'] = text
        res['text'] = res['text'].str.replace('-\n', '')
        res['text'] = res['text'].str.replace('\x0c', '')
        res['text'] = res['text'].str.split('\n\n')
        res['text'] = res['text'].apply(process_text_list)
        res['text'] = res['text'].str.replace('————', '')
        res['text'] = res['text'].str.replace(r"[|;@{}™=]|(—_)|(\«)|(o:)|[_~/<¢:°!]|[—]{2,}", '')
        res['text'] = res['text'].str.replace(r"]|\[|\(|\)|-|(\\){1,}", '')
        res['text'] = res['text'].str.replace(r"[ ]{2,}", ' ')
        
        for i in res.iterrows():
            if len(i[1]['text']) < 5:
                res.drop(i[0], axis = 0, inplace = True)
        res.reset_index(inplace = True, drop = True)
        
        # Adding text embedings in the new column
        res['emb'] = res['text'].apply(lambda x: txt_model.encode(x))
        
        return image, n_image, res

def return_closest_avg_wh(result_df, box_idx, avg_width, avg_height):
    '''
    This function takes in the result dataframe and then based on the provided box id, will predict a single closest/most textually similar box if it exists
    Some print commands are there for debugging
    '''
    sel_box = result_df.loc[box_idx]
    result_df = result_df.copy()
    
    result_df = result_df.loc[box_idx:]
    
    result_df['x_distance'] = abs(result_df['x_centers'] - result_df.loc[box_idx, 'x_centers'])
    result_df['y_distance'] = abs(result_df['y_centers'] - result_df.loc[box_idx, 'y_centers'])
    
    temp = result_df[result_df['x_distance'] <= 1.3 * avg_width]
    temp = temp[temp['y_distance'] <= avg_height]
    
    if len(temp) > 1:
        # Calculating the similarity score
        txt0 = temp.loc[box_idx]['emb']
        temp['sim'] = temp['emb'].apply(lambda x: pytorch_cos_sim(txt0, x).item())
        temp = temp.sort_values('sim', ascending = False)
        temp = temp.drop(box_idx, axis = 0)
        print(temp)
        return temp.iloc[[0]]
    else:
        return None

def create_new_order_jumps(image, result_df, jumps):
    '''
    result_df: provide the original result dataframe
    threshold: not being used currently, but can be used to define a minimum text similarity threshold for stuff

    Outputs:
    Creates a temp dataframe that has the correct order. At least its supposed to have correct order
    '''
    temp = pd.DataFrame()
    result_df = result_df.copy()
    og_result_df = result_df.copy()

    result_df['width'] = result_df['xmax'] - result_df['xmin']
    result_df['height'] = result_df['ymax'] - result_df['ymin']
    
    # avg_width = np.average(result_df['width'], weights = result_df['width']/np.sum(result_df['width']))
    avg_width = np.mean(result_df['width'])
    # avg_height = np.average(result_df['height'], weights = result_df['height']/np.sum(result_df['height']))
    avg_height = np.max(result_df['y_centers']) - np.min(result_df['y_centers'])

    while len(result_df) >= 1:
        idx = []
        temp = pd.concat([temp, result_df.iloc[[0]]], axis = 0)
        idx.append(temp.iloc[[-1]].index.values[0])

        for i in range(jumps):
            next_box = return_closest_avg_wh(result_df, temp.iloc[[-1]].index.values[0], avg_width, avg_height)
            temp = pd.concat([temp, next_box])
            idx.append(temp.iloc[[-1]].index.values[0])
            if next_box is None:
                break

        # print('JUMP')
        result_df.drop(idx, inplace = True)
    temp = temp.reset_index(drop = True)

    # Plotting the image order
    # if plot:
    plt.figure(figsize = (10, 30))
    # plt.imshow(plot_order(image, og_result_df))
    plt.imshow(plot_order(image, temp))
    plt.show()
    return temp

def pipeline_new_image_text(img_path, model_file, text_model, conf, threshold, jumps, save_path, columnar_data):
    '''
    img_path: Can be a list of image paths or a single image
    model_file: Variable with the loaded model for Object detection
    text_model: The model for text similarity estimation
    conf: Confidence level for prediction of boxes
    threshold: Threshold value for merging of overlapping boxes, value between 0-1
    jumps: Number of jumps from box-to-box, default value is 2 but can be changed based on the image
    save_path: Path of the folder that should contain the result .txt files and the images showing the order
    columnar_data: If the data in the image is basically supposed to be read top to bottom in each column, default is False
    '''
    if type(img_path) == list:
        for i in img_path:
            print(i)
            img, _, res = model_inference_full_size(i, model_file, text_model, conf, threshold)
            if res is not None:
                if columnar_data == False:
                    res = create_new_order_jumps(img, res, jumps)

                name = i.split('\\')[-1][:-4]
                with open(save_path + '/' + name + '.txt', 'w') as f:
                    f.write('\n\n'.join(res['text']))
                print(f'Text for {name} saved')

                cv2.imwrite(save_path + '/' + name + '.png', plot_order(img, res))
    else:
        img, _, res = model_inference_full_size(img_path, model_file, text_model, conf, threshold)

        if res is not None:
            if columnar_data == False:
                res = create_new_order_jumps(img, res, jumps)

            name = img_path.split('\\')[-1][:-4]
            with open(save_path + '/' + name + '.txt', 'w') as f:
                f.write('\n\n'.join(res['text']))
            print(f'Text for {name} saved')

            cv2.imwrite(save_path + '/' + name + '.png', plot_order(img, res))

# Loading the trained Pytorch model
model = torch.hub.load('yolov5', 'custom', path = model_file_path, source = 'local')

# Loading the trained Sentence similarity model
txt_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

all_paths = glob.glob(f'{input_path}/*.*')
pipeline_new_image_text(r'all_jp2\rough41.jp2', model, txt_model, 0.01, 0.3, jumps = jumps, save_path = res_path, columnar_data = column_data)