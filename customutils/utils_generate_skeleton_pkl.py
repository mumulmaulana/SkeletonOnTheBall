import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle
import os

def format_annotation(skeleton_file, filename, id, augmented, total_frames):
    keypoints_list = []
    scores_list = []

    for frame in range(15, 36):
        if id == None: # if id is None, that means the player is on the ball
            if augmented == None:
                row = skeleton_file[(skeleton_file.iloc[:, 0] == frame) & (skeleton_file.iloc[:, 2] == 1)]
            else:
                if augmented == 'mirror':
                    row = skeleton_file[(skeleton_file.iloc[:, 0] == frame) & (skeleton_file.iloc[:, 1] == 90)]
                elif augmented == 'scale':
                    row = skeleton_file[(skeleton_file.iloc[:, 0] == frame) & (skeleton_file.iloc[:, 1] == 91)]
                elif augmented == 'translate':
                    row = skeleton_file[(skeleton_file.iloc[:, 0] == frame) & (skeleton_file.iloc[:, 1] == 92)]
        else:
            row = skeleton_file[(skeleton_file.iloc[:, 0] == frame) & (skeleton_file.iloc[:, 1] == id)] 

        if row.empty: # debugging purpose
            print(filename, frame, id, augmented)

        if augmented != None:
            row = row.drop(columns=row.columns[3:7]).reset_index(drop=True)
            row.columns = range(row.shape[1]) # Reset the column indices

        keypoints = []
        scores = []
        for i in range(17):
            y, x, confidence = row.iloc[0, 3 + i*3], row.iloc[0, 4 + i*3], row.iloc[0, 5 + i*3]
            keypoints.append([x, y])
            scores.append(confidence)
        
        keypoints_list.append(keypoints)
        scores_list.append(scores)

    keypoints_list_np = np.array(keypoints_list)
    scores_list_np = np.array(scores_list)
    keypoints_list_np = np.expand_dims(keypoints_list_np, axis=0)  # Add one more dimension
    scores_list_np = np.expand_dims(scores_list_np, axis=0)  # Add one more dimension

    anno = dict()
    anno['keypoint'] = keypoints_list_np
    anno['keypoint_score'] = scores_list_np
    anno['img_shape'] = (720, 1280)
    anno['original_shape'] = (720, 1280)
    anno['total_frames'] = total_frames
    anno['frame_dir'] = filename

    if id == None: # if id is None, that means the player is on the ball
        anno['label'] = 1
    else:
        anno['label'] = 0

    return anno

def format_pkl_structure(filename, split, split_type):
    if filename.split('_')[-2] == 'off':
        id = int(filename.split('_')[-1])
        action = filename.split('_')[-3]
        filename = '_'.join(filename.split('_')[:-3])
        augmented = None
    else:
        id = None
        if filename.split('_')[-1] in ['mirror', 'scale', 'translate']:
            action = filename.split('_')[-2]
            augmented = filename.split('_')[-1]
            filename = '_'.join(filename.split('_')[:-2])
        else:
            action = filename.split('_')[-1]
            augmented = None
            filename = '_'.join(filename.split('_')[:-1])
    
    if augmented == None:
        skeleton_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), filename+'.csv'), header=None)
    else:
        skeleton_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), filename+'_augmented.csv'), header=None)

    if id == None: # if id is None, that means the player is on the ball
        if augmented == None:
            filename_anno = '{}_{}_on'.format(filename, action)
        else:
            filename_anno = '{}_{}_on_{}'.format(filename, action, augmented)
    else:
        filename_anno = '{}_{}_off_{}'.format(filename, action, id)
    split[split_type].append(filename_anno)

    return split, skeleton_file, filename_anno, id, augmented

# Load the JSON files
with open('/media/ogatalab/OgataLab8TB/captured_frame/on_the_ball_21.json', 'r') as f:
    on_the_ball_data = json.load(f)

with open('/media/ogatalab/OgataLab8TB/captured_frame/off_the_ball_21.json', 'r') as f:
    off_the_ball_data = json.load(f)

# Convert JSON data to DataFrames
df_on_the_ball = pd.DataFrame(on_the_ball_data)
df_off_the_ball = pd.DataFrame(off_the_ball_data)

# Add a label column
df_on_the_ball['label'] = 1
df_off_the_ball['label'] = 0

# Combine the DataFrames
df = pd.concat([df_on_the_ball, df_off_the_ball, df_on_the_ball], ignore_index=True)

# Features and labels
X = df.drop(columns=['label'])
y = df['label']

# Custom split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42, stratify=y_rem)

# Combine the training and validation sets
X_crossval = pd.concat([X_train, X_val], ignore_index=True)
y_crossval = pd.concat([y_train, y_val], ignore_index=True)

# Initialize StratifiedKFold
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
class_weights_list = []

for fold, (train_index, test_index) in enumerate(skf.split(X_crossval, y_crossval)):
    X_train, X_val = X_crossval.iloc[train_index], X_crossval.iloc[test_index]
    y_train, y_val = y_crossval.iloc[train_index], y_crossval.iloc[test_index]
    
    print(len(X_train), len(X_val))

    # Compute class weights for the training data
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights for Fold {fold + 1}: {class_weights_dict}")
    class_weights_list.append(class_weights_dict)

    # for augmentation in ['mirror', 'scale', 'translate']: # disable this line when running normal version

    split = {'train': [], 'val': [], 'test': []}
    annotations = []

    print("Start pickle formatting for Test set of Fold {}".format(fold + 1))
    for index, row in X_test.iterrows():
        filename = row[0]
        split, skeleton_file, filename_anno, id, augmented = format_pkl_structure(filename, split, 'test')
        # if augmented != None: augmented = augmentation # disable this line when running normal version
        anno = format_annotation(skeleton_file, filename_anno, id, augmented, 21)
        annotations.append(anno)

    print("Start pickle formatting for Train set of Fold {}".format(fold + 1))
    for index, row in X_train.iterrows():
        filename = row[0]
        split, skeleton_file, filename_anno, id, augmented = format_pkl_structure(filename, split, 'train')
        # if augmented != None: augmented = augmentation # disable this line when running normal version
        anno = format_annotation(skeleton_file, filename_anno, id, augmented, 21)
        annotations.append(anno)

    print("Start pickle formatting for Validation set of Fold {}".format(fold + 1))
    for index, row in X_val.iterrows():
        filename = row[0]
        split, skeleton_file, filename_anno, id, augmented = format_pkl_structure(filename, split, 'val')
        # if augmented != None: augmented = augmentation # disable this line when running normal version
        anno = format_annotation(skeleton_file, filename_anno, id, augmented, 21)
        annotations.append(anno)

    print("Dumping pkl_class1dbl_format_fold_{}.pkl".format(fold + 1))
    pkl = dict()
    pkl['split'] = split
    pkl['annotations'] = annotations
    with open('pkl_class1dbl_format_fold_{}.pkl'.format(fold + 1), 'wb') as f:
        pickle.dump(pkl, f)

# Save the class weights
with open('class_weights_class1dbl.txt', 'w') as f:
    for i, class_weight in enumerate(class_weights_list):
        f.write(f"Fold {i + 1}: {class_weight}\n")