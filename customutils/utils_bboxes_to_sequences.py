import os
import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
from tqdm import tqdm

# local modules
sys.path.insert(0, '/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/SkeletonOnTheBall')
from annotate.module.simpleHRNet.misc.visualization import joints_dict, get_points_and_skeleton_color, draw_id, draw_bbox, draw_points_and_skeleton_v2

def get_csv_file(frame_format):
    if frame_format.split('_')[-2] in ['on', 'off']:
        if frame_format.split('_')[-2] == 'off': # ex: '..._off_1'
            id = int(frame_format.split('_')[-1])
        else: # ex: '..._on_mirror'
            id = None
        csv_name = '_'.join(frame_format.split('_')[:-2])
    else: # ex: '..._on'
        id = None
        csv_name = '_'.join(frame_format.split('_')[:-1])

    if csv_name.split('_')[-1] in ['Shoot', 'Pass', 'Dribble']:
        action = csv_name.split('_')[-1]
        csv_name = '_'.join(csv_name.split('_')[:-1])
        skeleton_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), csv_name+'.csv'), header=None)
    else:
        for action in ['Shoot', 'Pass', 'Dribble']:
            if os.path.exists(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), csv_name+'.csv')):
                skeleton_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), csv_name+'.csv'), header=None)
                break
    
    return csv_name, skeleton_file, action, id

def from_frame_dir_format(frame_dir):
    csv_name, skeleton_file, action, id = get_csv_file(frame_dir)
    skeleton_file = skeleton_file.rename(columns={skeleton_file.columns[0]: 'frame', skeleton_file.columns[1]: 'player_id', skeleton_file.columns[2]: 'in_action'})
    bbox_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), csv_name+'_bboxes.csv'))
    bbox_file['frame'] = skeleton_file['frame'].values
    bbox_file['in_action'] = skeleton_file['in_action'].values
    
    video_file = '/media/ogatalab/OgataLab8TB/captured_frame/'+(action if action in ('Shoot', 'Pass', 'Dribble') else 'test_videos')+'/'+csv_name+'.mp4'
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {video_file}")

    for frame in range(15, 36):
        if id == None: # if id is None, that means the player is on the ball
            skeleton = skeleton_file.loc[(skeleton_file['frame'] == frame) & (skeleton_file['in_action'] == 1)].iloc[:, 3:].to_numpy().reshape(-1, 17, 3)
            x1, y1, x2, y2 = bbox_file.loc[(bbox_file['frame'] == frame) & (bbox_file['in_action'] == 1)].iloc[:, :4].to_numpy()[0]
        else:
            skeleton = skeleton_file.loc[(skeleton_file['frame'] == frame) & (skeleton_file['player_id'] == id)].iloc[:, 3:].to_numpy().reshape(-1, 17, 3)
            x1, y1, x2, y2 = bbox_file.loc[(bbox_file['frame'] == frame) & (bbox_file['player_id'] == id)].iloc[:, :4].to_numpy()[0]
        # Ensure that none of x1, y1, x2, y2 are less than 0
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = max(x2, 0)
        y2 = max(y2, 0)

        # set for raw, overlay, and blank
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, image = cap.read()
        skeleton_img = image.copy()
        blank = np.ones((720, 1280, 3), dtype=np.uint8)
        blank *= 255 # set to white
        
        skeleton_img = draw_points_and_skeleton_v2(skeleton_img, skeleton[0], joints_dict()['coco']['skeleton'], (255, 0, 0), (0, 0, 255), confidence_threshold=0.3)
        skeleton_blank = draw_points_and_skeleton_v2(blank, skeleton[0], joints_dict()['coco']['skeleton'], (255, 0, 0), (0, 0, 255), confidence_threshold=0.3)
        
        cropped_raw = image[y1:y2, x1:x2]
        cropped_overlay = skeleton_img[y1:y2, x1:x2]
        cropped_blank = skeleton_blank[y1:y2, x1:x2]

        # save the image
        for mode, cropped_player in zip(['raw', 'overlay', 'blank'], [cropped_raw, cropped_overlay, cropped_blank]):
            if not os.path.exists('/media/ogatalab/OgataLab8TB/captured_frame/crop_skeleton/'+frame_dir):
                os.makedirs('/media/ogatalab/OgataLab8TB/captured_frame/crop_skeleton/'+frame_dir)
            cv2.imwrite('/media/ogatalab/OgataLab8TB/captured_frame/crop_skeleton/'+frame_dir+'/'+mode+'_'+str(frame)+'.jpg', cropped_player)

    cap.release()

def crop_and_save(cropped_folder, csv_name, skeleton_file, action, id, flip=False):
    if not os.path.exists('/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/imgs/'+cropped_folder):
        os.makedirs('/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/imgs/'+cropped_folder)
        skeleton_file = skeleton_file.rename(columns={skeleton_file.columns[0]: 'frame', skeleton_file.columns[1]: 'player_id', skeleton_file.columns[2]: 'in_action'})
        bbox_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), csv_name+'_bboxes.csv'))
        bbox_file['frame'] = skeleton_file['frame'].values
        bbox_file['in_action'] = skeleton_file['in_action'].values
        
        video_file = '/media/ogatalab/OgataLab8TB/captured_frame/'+(action if action in ('Shoot', 'Pass', 'Dribble') else 'test_videos')+'/'+csv_name+'.mp4'
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            for frame in range(15, 36):
                if id == None: # if id is None, that means the player is on the ball
                    x1, y1, x2, y2 = bbox_file.loc[(bbox_file['frame'] == frame) & (bbox_file['in_action'] == 1)].iloc[:, :4].to_numpy()[0]
                else:
                    x1, y1, x2, y2 = bbox_file.loc[(bbox_file['frame'] == frame) & (bbox_file['player_id'] == id)].iloc[:, :4].to_numpy()[0]
                    
                # Ensure that none of x1, y1, x2, y2 are less than 0
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = max(x2, 0)
                y2 = max(y2, 0)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, image = cap.read()
                if ret:
                    cropped_player = image[y1:y2, x1:x2]
                    if flip: cropped_player = cv2.flip(cropped_player, 1)
                    # plt.imshow(cropped_player, cmap = 'gray', interpolation = 'bicubic')
                    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                    # plt.show()
                    cv2.imwrite('/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/imgs/'+cropped_folder+'/'+str(frame)+'.jpg', cropped_player)
                else:
                    print(f'\nError reading frame {frame} from {video_file}')
                    break
            cap.release()

def from_pkl_format():
    for i in range(1,11):
        pkl_name = 'pkl_format_fold_{}_new'.format(i)
        pkl_path = '/media/ogatalab/OgataLab8TB/captured_frame/'+pkl_name+'.pkl'
        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)

        print('Processing {}'.format(pkl_name))
        split = pkl_data['split']
        for key, value in split.items():
            print('{} split'.format(key))
            total = len(value)
            cropped_list = {'img_sequences': [], 'labels': []}
            with tqdm(total=total, desc="Starting") as pbar:
                for iter, element in enumerate(value):
                    csv_name, skeleton_file, action, id = get_csv_file(element)
                    
                    if id == None: # if id is None, that means the player is on the ball
                        cropped_folder = csv_name+'_'+action+'_on'
                        cropped_list['labels'].append(1)
                    else:
                        cropped_folder = csv_name+'_'+action+'_off_'+str(id)
                        cropped_list['labels'].append(0)
                    cropped_list['img_sequences'].append(cropped_folder)

                    pbar.set_description(f"({iter+1}/{total}) Processing {cropped_folder}")

                    # create duplicate for mirror data (comment if data is already mirrored)
                    if id == None:
                        cropped_folder_mirror = csv_name+'_'+action+'_on_mirror'
                        cropped_list['labels'].append(1)
                        cropped_list['img_sequences'].append(cropped_folder_mirror)

                        pbar.set_description(f"({iter+1}/{total}) Processing {cropped_folder_mirror}")

                        crop_and_save(cropped_folder, csv_name, skeleton_file, action, id, flip=True)

                    pbar.update(1)
                    
                    crop_and_save(cropped_folder, csv_name, skeleton_file, action, id)

            if not os.path.exists('/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/'+pkl_name+'_augmented'):
                os.makedirs('/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/'+pkl_name+'_augmented')
            with open('/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/'+pkl_name+'_augmented/'+key+'.pkl', 'wb') as f:
                pickle.dump(cropped_list, f)
        
        print('\n')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('frame_dir', help='Path to the image (skeleton-model format)')
    args = args.parse_args()

    if args.frame_dir == '':
        from_pkl_format()
    else:
        from_frame_dir_format(args.frame_dir)