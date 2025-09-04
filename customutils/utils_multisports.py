import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils_skeleton_plot import normalize_skeletons

def get_splits(pkl_data, save_path):
    annos = pkl_data['annotations']
    X = []
    y = []
    for anno in annos:
        video_name = anno['frame_dir']
        # print(video_name)
        if '_mirror' not in video_name:
            X.append(video_name)
            if '_shoot' in video_name:
                y.append(3)
            elif '_dribble' in video_name:
                y.append(2)
            elif '_pass' in video_name:
                y.append(1)

    # separate splits for train_val_test and all_train
    train_val_test_split = {'train': [], 'val': [], 'test': []}
    all_train_split = {'train': [], 'val': [], 'test': []}
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    X_train, X_rem, _, y_rem = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42, stratify=y)
    X_val, X_test, _, _ = train_test_split(X_rem, y_rem, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42, stratify=y_rem)
    
    # add mirror data to the splits (comment if data is already mirrored)
    X_train_mirror = [video_name+'_mirror' for video_name in X_train]
    X_val_mirror = [video_name+'_mirror' for video_name in X_val]
    X_test_mirror = [video_name+'_mirror' for video_name in X_test]
    X_mirror = [video_name+'_mirror' for video_name in X]
    X_train.extend(X_train_mirror)
    X_val.extend(X_val_mirror)
    X_test.extend(X_test_mirror)
    X.extend(X_mirror)
    
    train_val_test_split['train'] = X_train
    train_val_test_split['val'] = X_val
    train_val_test_split['test'] = X_test
    all_train_split['train'] = X
    # print(f'train_val_test split: {len(X_train)}, {len(X_val)}, {len(X_test)}')
    # print(f'all_train split: {len(X)}')

    # save two version of pkl files
    train_val_test = {'split': train_val_test_split, 'annotations': annos}
    all_train = {'split': all_train_split, 'annotations': annos}

    with open(os.path.join(save_path, 'mmformat_trainvaltest.pkl'), 'wb') as f:
        pickle.dump(train_val_test, f)
    with open(os.path.join(save_path, 'mmformat_alltrain.pkl'), 'wb') as f:
        pickle.dump(all_train, f)

def append_to_soccernet(pkl_path, target_pkl_path):
    with open(pkl_path, 'rb') as file:
        pkl_data = pickle.load(file)
    with open(target_pkl_path, 'rb') as file:
        target_data = pickle.load(file)
    save_path = os.path.dirname(target_pkl_path)
    
    # append to the target data
    annos = pkl_data['annotations']
    target_annos = target_data['annotations']
    
    # Combine the arrays
    combined = target_annos + annos

    # Determine target temporal dimension (e.g., min T)
    target_T = min(d['keypoint'].shape[1] for d in combined)

    # Normalize all 'keypoint' arrays to the target_T dimension
    for d in combined:
        _, T, V, C = d['keypoint'].shape
        if T > target_T:
            start_idx = (T - target_T) // 2
            end_idx = start_idx + target_T
            d['keypoint'] = d['keypoint'][:, start_idx:end_idx, :, :]
            d['keypoint_score'] = d['keypoint_score'][:, start_idx:end_idx, :]
            d['total_frames'] = target_T

    print(len(combined))
    normalized = normalize_skeletons(combined, (1280, 720))
    splits = pkl_data['split']
    target_splits = target_data['split']
    for key in target_splits.keys():
        print(len(target_splits.get(key)))
        if len(splits.get(key)) > 0:
            target_splits[key].extend(splits.get(key))
    print(len(target_splits.get('train')))
    print(len(target_splits.get('val')))
    print(len(target_splits.get('test')))

    # save the new pkl file
    new_pkl_data = {'split': target_splits, 'annotations': normalized}
    with open(os.path.join(save_path, 'pkl_format_fold_1_new_2class_multisports_alltrain.pkl'), 'wb') as f:
        pickle.dump(new_pkl_data, f)
    
def extend_gt_to_csv(pkl_data, csv_path):
    total = len(pkl_data)
    with tqdm(total=total) as pbar:
        for iter, (key, value) in enumerate(pkl_data.items()): # key: video name, value: n x [video_name, frame_start (for csv indexing), frame_num, x1, y1, x2, y2]
            pbar.update(1)
            pbar.set_description(f"({iter+1}/{total}) Processing {key}")

            gt_bboxes = value[:, 3:] # shape: (n, 4)
            # Combine the x and y coordinates to form the centroids array
            centroid_x_coords = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2).astype(int)
            centroid_y_coords = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2).astype(int)
            gt_centroids = np.column_stack((centroid_x_coords, centroid_y_coords))  # Shape: (n, 2)
            frame_start = value[0, 1] # first row because the value
            offset = int(value[0, 2] - frame_start) # frame_num - frame_start
            frame_to_process = np.arange(1, 22) if len(value) < 21 else np.arange(1, len(value)+1)
            frame_to_process = frame_to_process[offset:offset+len(value)]

            csv_bbox_data = pd.read_csv(os.path.join(csv_path, key+'_bboxes.csv'))
            csv_skeleton_data = pd.read_csv(os.path.join(csv_path, key+'.csv'), header=None)
            csv_skeleton_data = csv_skeleton_data.rename(columns={csv_skeleton_data.columns[0]: 'frame', csv_skeleton_data.columns[1]: 'player_id', csv_skeleton_data.columns[2]: 'in_action'})
            csv_bbox_data['frame'] = csv_skeleton_data['frame'].values.astype(int)

            first_last_frame = [0, 0] # to handle the case where the player is not found in the csv file and extending is needed
            first_last_ids = [0, 0]
            store_first = True
            for i, frame_num in enumerate(frame_to_process):
                csv_bboxes = csv_bbox_data.loc[(csv_bbox_data['frame'] == frame_num)].iloc[:, :4].to_numpy()
                centroid_x_coords = ((csv_bboxes[:, 0] + csv_bboxes[:, 2]) / 2).astype(int)
                centroid_y_coords = ((csv_bboxes[:, 1] + csv_bboxes[:, 3]) / 2).astype(int)
                csv_centroids = np.column_stack((centroid_x_coords, centroid_y_coords))  # Shape: (n, 2)
                if len(csv_centroids) == 0:
                    continue

                centroid_diffs = np.linalg.norm(np.expand_dims(gt_centroids[i], axis=0) - csv_centroids, axis=1)
                # if key == 'v_IOeGNAMz2ek_c003_dribble_1':
                #     print(centroid_diffs, csv_centroids, gt_centroids[i], frame_num)
                best_match_idx = np.argmin(centroid_diffs)

                player_id = csv_bbox_data.loc[(csv_bbox_data['frame'] == frame_num)].iloc[best_match_idx]['player_id']
                csv_skeleton_row_idx = csv_skeleton_data.loc[(csv_skeleton_data['frame'] == frame_num) & (csv_skeleton_data['player_id'] == player_id)].index
                csv_skeleton_data.loc[csv_skeleton_row_idx, 'in_action'] = 1

                first_last_frame[1] = frame_num
                first_last_ids[1] = player_id
                if store_first:
                    first_last_frame[0] = frame_num
                    first_last_ids[0] = player_id
                    store_first = False
                
            # comment if extending is not needed
            if first_last_frame[0] != 0 and len(frame_to_process) < 21:
                backwards = first_last_frame[0] - 1
                while backwards > 0:
                    csv_skeleton_row = csv_skeleton_data.loc[(csv_skeleton_data['frame'] == backwards) & (csv_skeleton_data['player_id'] == first_last_ids[0])]
                    if len(csv_skeleton_row) > 0:
                        csv_skeleton_row_idx = csv_skeleton_row.index
                        csv_skeleton_data.loc[csv_skeleton_row_idx, 'in_action'] = 1
                    backwards -= 1
                forwards = first_last_frame[1] + 1
                while forwards <= 21:
                    csv_skeleton_row = csv_skeleton_data.loc[(csv_skeleton_data['frame'] == forwards) & (csv_skeleton_data['player_id'] == first_last_ids[1])]
                    if len(csv_skeleton_row) > 0:
                        csv_skeleton_row_idx = csv_skeleton_row.index
                        csv_skeleton_data.loc[csv_skeleton_row_idx, 'in_action'] = 1
                    forwards += 1

            csv_skeleton_data.to_csv(os.path.join(csv_path, key+'.csv'), index=False, header=False)

if __name__ == "__main__":
    csv_path = '/media/ogatalab/OgataLab8TB/MultiSports/extracted_videos/len_original'
    # file_path = '/media/ogatalab/OgataLab8TB/MultiSports/football_selected.pkl'
    # file_path = '/media/ogatalab/OgataLab8TB/MultiSports/extracted_frames/len_15/annotations.pkl'
    file_path = '/media/ogatalab/OgataLab8TB/MultiSports/extracted_frames/len_15/mmformat_alltrain.pkl'
    target_path = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror_normalized.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # extend_gt_to_csv(data, csv_path)
    # get_splits(data, '/media/ogatalab/OgataLab8TB/MultiSports/extracted_frames/len_15')
    append_to_soccernet(file_path, target_path)
