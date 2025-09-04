import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.lines import Line2D

# Replace 'your_file.pkl' with the path to your .pkl file
# file_path = '2015-02-21-17-30Paderborn0-6BayernMunich1_720p_16897.pkl'
file_path_1 = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror.pkl'
# file_path_2 = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_all.pkl'
file_path_2 = '/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/vggtcn_fold_1/test.pkl'

# Open the .pkl file in binary read mode
with open(file_path_1, 'rb') as file_1:
    data_1 = pickle.load(file_1)
with open(file_path_2, 'rb') as file_2:
    data_2 = pickle.load(file_2)

# Check if the data is a dictionary
# if isinstance(data, dict):
#     for key, value in data.items():
#         if key == 'split':
#             for k, v in value.items():
#                 print(f"length of {k}: {len(v)}")
#         else:
#             for k, v in enumerate(value):
#                 print(v)
#                 break
#             print(f"length of {key}: {len(value)}")
# else:
#     print("The loaded data is not a dictionary")

def get_train_weights(data, file_path):
    if isinstance(data, dict):
        annos = data['annotations']
        annos_indexed = {anno['frame_dir']: anno for anno in annos}
        train_split = data['split']['train']
        val_split = data['split']['val']
        test_split = data['split']['test']
        labels = {'train': [], 'val': [], 'test': []}
        labels['train'] = [anno['label'] for anno in annos_indexed.values() if anno['frame_dir'] in train_split]
        labels['val'] = [anno['label'] for anno in annos_indexed.values() if anno['frame_dir'] in val_split]
        labels['test'] = [anno['label'] for anno in annos_indexed.values() if anno['frame_dir'] in test_split]
        unique_labels = np.unique(labels['train'])
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels['train'])
        class_weights_dict = dict(zip(unique_labels, class_weights))
        
        for split in ['train', 'val', 'test']:
            for label in unique_labels:
                print(f"({split}) Number of {label}: {labels[split].count(label)}")
        print(f"Class weights for {file_path}: {class_weights_dict}")

        return class_weights_dict
    else:
        print("The loaded data is not a dictionary.")

def flip_anno_coords(data, file_path):
    if isinstance(data, dict):
        annos = data['annotations']
        for i, anno in enumerate(annos):
            for j, person in enumerate(anno['keypoint']):
                for k, frame in enumerate(person):
                    flipped_joints = frame[:, [1, 0]] # flip [y, x] to [x, y]
                    data['annotations'][i]['keypoint'][j][k] = flipped_joints

        filename = file_path
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        print("The loaded data is not a dictionary.")

def convert_4class_to_2class(file_path):
    if file_path.endswith('4class.pkl'):
        class_weights_list = []
        for fold in range(1, 11):
            filepath = file_path.replace('_1_', f'_{fold}_')
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            print(f"Converting {filepath} to 2-class format.")
            if isinstance(data, dict):
                annos = data['annotations']
                y = []
                for i, anno in enumerate(annos):
                    if anno['label'] == 2 or anno['label'] == 3:
                        data['annotations'][i]['label'] = 1

                # get y_train
                train = data['split']['train']
                for element in train:
                    if '_on' in element:
                        y.append(1)
                    else:
                        y.append(0)

                # Compute class weights for the training data
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
                class_weights_dict = dict(zip(np.unique(y), class_weights))
                print(f"Class weights for Fold {fold}: {class_weights_dict}")
                class_weights_list.append(class_weights_dict)

                filename = filepath.split('/')[-1]
                filename = filename.replace('4class', '2class')
                with open(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format', filename), 'wb') as f:
                    pickle.dump(data, f)
            else:
                print("The loaded data is not a dictionary.")

                        
        with open(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format', 'class_weights_2class.txt'), 'w') as f:
            for i, class_weight in enumerate(class_weights_list):
                f.write(f"Fold {i + 1}: {class_weight}\n")
    else:
        print("The file is not a 4-class .pkl file.")

def convert_split_to_cnnlstm_format(data, save_path):
    splits = data['split']

    for key, value in splits.items():
        pkl_dict = {'img_sequences': [], 'labels': []}
        for element in value:
            pkl_dict['img_sequences'].append(element)
            if '_on' in element:
                pkl_dict['labels'].append(1)
            else:
                pkl_dict['labels'].append(0)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, key+'.pkl'), 'wb') as f:
            pickle.dump(pkl_dict, f)

def get_class_split(data, mode):
    train = data['split']['train']
    val = data['split']['val']
    test = data['split']['test']
    annos = data['annotations']
    index_table = {anno['frame_dir']: idx for idx, anno in enumerate(annos)}
    X = []
    y = []

    for iter in [train, val, test]:
        for element in iter:
            X.append(element)
            if 'off' in element:
                y.append(0)
            else:
                if mode == '2class':
                    y.append(1)
                    annos[index_table[element]]['label'] = 1
                else:
                    if '_Pass' in element or '_pass' in element:
                        y.append(1)
                        annos[index_table[element]]['label'] = 1
                    elif '_Dribble' in element or '_dribble' in element:
                        y.append(2)
                        annos[index_table[element]]['label'] = 2
                    elif '_Shoot' in element or '_shoot' in element:
                        y.append(3)
                        annos[index_table[element]]['label'] = 3
                    else:
                        raise ValueError("Cannot determine the class of the element.")

    ret = dict()
    ret['X'] = X
    ret['y'] = y
    ret['annos'] = annos
    return ret

def generate_train_val_test(data, file_path, mode='2class'):
    if isinstance(data, dict):
        class_split = get_class_split(data, mode)
        # print('Anns[0] key: {}'.format(next(iter(class_split['annos']))))
        # raise ValueError("Stop here.")

        # Custom split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        X_train, X_rem, y_train, y_rem = train_test_split(class_split['X'], class_split['y'], test_size=(1 - train_ratio), random_state=42, stratify=class_split['y'])
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42, stratify=y_rem)

        if mode == '2class': # convert all non 0 class (Pass, Dribble, Shoot) to 1
            y_train = [0 if y == 0 else 1 for y in y_train]
            y_val = [0 if y == 0 else 1 for y in y_val]
            y_test = [0 if y == 0 else 1 for y in y_test]

        # combine train and val data
        X_crossval = X_train + X_val
        y_crossval = y_train + y_val

        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        class_weights_list = []
        for fold, (train_index, val_index) in enumerate(skf.split(X_crossval, y_crossval)):
            X_train = [X_crossval[i] for i in train_index]
            X_val = [X_crossval[i] for i in val_index]
            y_train = [y_crossval[i] for i in train_index]
            y_val = [y_crossval[i] for i in val_index]

            print(len(X_train), len(X_val))

            # Compute class weights for the training data
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = dict(zip(np.unique(y_train), class_weights))
            print(f"Class weights for Fold {fold + 1}: {class_weights_dict}")
            class_weights_list.append(class_weights_dict)

            data['split']['train'] = X_train
            data['split']['val'] = X_val
            data['split']['test'] = X_test
            data['annotations'] = class_split['annos']
            
            filename = os.path.splitext(file_path)[0]
            filename = filename.split('/')[-1]
            filename = filename.replace('_1_', f'_{fold+1}_')
            with open(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format', filename+'_'+mode+'.pkl'), 'wb') as f:
                pickle.dump(data, f)

        with open(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format', 'class_weights_'+mode+'.txt'), 'w') as f:
            for i, class_weight in enumerate(class_weights_list):
                f.write(f"Fold {i + 1}: {class_weight}\n")
        
    else:
        print("The loaded data is not a dictionary.")

def print_split_samples(data, file_path):
    print(f"\n{file_path}:")
    if isinstance(data, dict):
        split = data['split']
        annos = data['annotations']
        annos_indexed = {anno['frame_dir']: anno for anno in annos}
        for key, value in split.items():
            print(f"\n{key}:")
            print_flag = {'off': False, 'on': False, 'aug': False}
 
            for i, element in enumerate(value):  

                # basic sample print
                # if all(print_flag.values()): break
                # else:
                #     if not print_flag['off'] and '_off' in element:
                #         print_flag['off'] = True
                #         print(f"off: {element}")
                #     if not print_flag['on'] and '_on' in element:
                #         print_flag['on'] = True
                #         print(f"on: {element}")
                #     if not print_flag['aug'] and '_mirror' in element:
                #         print_flag['aug'] = True
                #         print(f"aug: {element}")

                # print the annotation pair of on-the-ball with its augmented version
                if '_mirror' in element:
                    get_aug = annos_indexed.get(element, 0)
                    print(f"aug: {get_aug['keypoint'][0][0][0]}")
                    get_on = annos_indexed.get(element.replace('_mirror', ''), 0)
                    print(f"on: {get_on['keypoint'][0][0][0]}")
                    break
    else:
        print("The loaded data is not a dictionary.")

def compare_class1_correct(file_path_1="", file_path_2=""):
    if file_path_1 == "": # skeleton-based model
        file_path_1 = '/home/ogatalab/Documents/mmaction2/work_dirs/stgcn/mirror/fold1/aagcn_2class_avgnormalized/test_result_class1.pkl'
        with open(file_path_1, 'rb') as file:
            data1 = pickle.load(file)
    if file_path_2 == "": # VGG model
        file_path_2 = '/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/vggtcn_fold_1/correct_class1.pkl'
        with open(file_path_2, 'rb') as file:
            data2 = pickle.load(file)

    # class1_correct from skeleton-based model and VGG model
    class1_correct_skeleton = [player['frame_dir'] for player in data1 if player['gt_label'] == 1 and player['pred_label'] == 1]
    class1_correct_vgg = [player['frame_dir'] for player in data2]
    print(f"Number of class1 correct in skeleton-based model: {len(class1_correct_skeleton)}")
    print(f"Number of class1 correct in VGG model: {len(class1_correct_vgg)}")

    # comparison of set elements
    print(f"Players in skeleton but not in VGG: {len(set(class1_correct_skeleton) - set(class1_correct_vgg))} players")
    for element in set(class1_correct_skeleton) - set(class1_correct_vgg):
        print(element)
    # print(f"Players in VGG but not in skeleton: {len(set(class1_correct_vgg) - set(class1_correct_skeleton))}")

def compare_to_cnnlstm_pkl(data_1, data_2, file_path_1, file_path_2):
    split_1 = data_1['split']['test']
    split_2 = data_2['img_sequences']

    if split_1 == split_2:
        print("The 'test' splits have the same elements in both files.")
    else:
        print("The 'test' splits have different elements in the files.")
        print(f"Elements in data_1['split']['test'] but not in data_2['img_sequences']: {len(set(split_1) - set(split_2))}")
        print(f"Elements in data_2['img_sequences'] but not in data_1['split']['test']: {len(set(split_2) - set(split_1))}")

# Function to compare 'split' values
def compare_splits(data_1, data_2, file_path_1, file_path_2):
    if isinstance(data_1, dict) and isinstance(data_2, dict):
        split_1 = data_1['split']
        split_2 = data_2['split']

        # comparison of set elements
        # if 'val' in split_1 and 'val' in split_2:
        #     test_1 = set(split_1['val'])
        #     test_2 = set(split_2['val'])
        #     if test_1 == test_2:
        #         print("The 'val' sets have the same elements in both files.")
        #     else:
        #         print("The 'val' sets have different elements in the files.")
        #         print(f"Elements in data_1['split']['val'] but not in data_2['split']['val']: {len(test_1 - test_2)}")
        #         print(f"Elements in data_2['split']['val'] but not in data_1['split']['val']: {len(test_2 - test_1)}")
        # else:
        #     print("'val' key is missing in one or both 'split' dictionaries.")

        # comparison of keypoint coordinates
        annos_1 = data_1['annotations']
        annos_2 = data_2['annotations']
        annos_1_indexed = {anno['frame_dir']: anno for anno in annos_1}
        annos_2_indexed = {anno['frame_dir']: anno for anno in annos_2}
        # annos_2_indexed = annos_2
        for key, value in split_1.items():
            print(f"\n{key}:")
            print_flag = {'off': False, 'on': False, 'aug': False}
            for element in value:
                if all(print_flag.values()): break
                else:
                    if not print_flag['off'] and '_off' in element:
                        print_flag['off'] = True
                        print(f"off: {element}")
                        get_anno1 = annos_1_indexed.get(element, 0)
                        get_anno2 = annos_2_indexed.get(element, 0)
                        print(f"{get_anno1['frame_dir']}: {get_anno1['keypoint'][0][0][0]} {get_anno1['label']}")
                        print(f"{get_anno2['frame_dir']}: {get_anno2['keypoint'][0][0][0]} {get_anno2['label']}")
                    elif not print_flag['on'] and '_on' in element:
                        print_flag['on'] = True
                        print(f"on: {element}")
                        get_anno1 = annos_1_indexed.get(element, 0)
                        get_anno2 = annos_2_indexed.get(element, 0)
                        print(f"{get_anno1['frame_dir']}: {get_anno1['keypoint'][0][0][0]} {get_anno1['label']}")
                        print(f"{get_anno2['frame_dir']}: {get_anno2['keypoint'][0][0][0]} {get_anno2['label']}")
                    elif not print_flag['aug'] and '_mirror' in element:
                        print_flag['aug'] = True
                        print(f"aug: {element}")
                        get_anno1 = annos_1_indexed.get(element, 0)
                        get_anno2 = annos_2_indexed.get(element, 0)
                        print(f"{get_anno1['frame_dir']}: {get_anno1['keypoint'][0][0][0]} {get_anno1['label']}")
                        print(f"{get_anno2['frame_dir']}: {get_anno2['keypoint'][0][0][0]} {get_anno2['label']}")

    else:
        print("One or both loaded data are not dictionaries.")

def check_for_duplicate(data, file_path):
    print(f"\n{file_path}:")
    if isinstance(data, dict):
        train = data['split']['train']
        val_dict = {element: element for element in data['split']['val']}
        test_dict = {element: element for element in data['split']['test']}
        class1_val = [element for element in val_dict if '_on' in element]
        class1_test = [element for element in test_dict if '_on' in element]
        val_count = 0
        test_count = 0
        for i, element in enumerate(train):
            if '_on' in element:
                if '_mirror' in element:
                    search_name = element.replace('_mirror', '')
                else:
                    search_name = element+'_mirror'

                get_val = val_dict.get(search_name, 0)
                get_test = test_dict.get(search_name, 0)
                val_count += 1 if get_val else 0
                test_count += 1 if get_test else 0

        print(f"Length of 'val': {len(val_dict)}")
        print(f"Number of class_1 in 'val': {len(class1_val)}")
        print(f"Number of duplicates in 'val': {val_count}")
        print(f"Length of 'test': {len(test_dict)}")
        print(f"Number of class_1 in 'test': {len(class1_test)}")
        print(f"Number of duplicates in 'test': {test_count}")

    else:
        print("The loaded data is not a dictionary.")

# Function to print the number of data for each class
def count_classes(data):
    if isinstance(data, dict):
        # Check the first augmented annotation
        # annos = data['annotations']
        # for anno in annos:
        #     if anno['frame_dir'].split('_')[-2] == 'on':
        #         print(anno)
        #         break

        # Count the number of data for each class
        if 'split' in data:
            split = data['split']
            # total_data = 0
            for key, value in split.items():
                print(f"\n{key}:")
                items = value
                class_count = {'class_0': 0, 'class_1': 0}
                for element in items:
                    # TO DO: determine classes based on frame name
                    if element.split('_')[-2] == 'off':
                        class_count['class_0'] += 1
                    else:
                        class_count['class_1'] += 1
                print("Number of data for each class:")
                print(f"Total data: {len(items)}")
                # total_data += len(items)
                for k, v in class_count.items():
                    print(f"Class {k}: {v}")
        # print('\n'+str(len(data['annotations']))+'\n'+str(total_data))
        # print('\n')
        # print(data['annotations'][0])
    else:
        print("The loaded data is not a dictionary.")

def multisport_pkl(data):
    if isinstance(data, dict):
        shoot_idxs = [33]
        pass_idxs = [34, 35, 36, 37]
        dribble_idxs = [38, 39]
        shoots = []
        passes = []
        dribbles = []
        
        labels = data['labels']
        for i, label in enumerate(labels):
            if i in shoot_idxs or i in pass_idxs or i in dribble_idxs: print(f"{i}: {label}")

        gttubes = data['gttubes']
        flag = False
        for key, value in gttubes.items():
            # print(f"{key}: {(value)}")
            # break
            for key, action in value.items():
                # print(f"{key}: {action}")
                # break
                if key in shoot_idxs:
                    for i, frame in enumerate(action):
                        # print(f"Length: {len(frame)}")
                        # print(f"{key}: {frame}")
                        # flag = True
                        # break
                        if len(frame) >= 10: shoots.append(len(frame))
                elif key in pass_idxs:
                    for i, frame in enumerate(action):
                        if len(frame) >= 10: passes.append(len(frame))
                elif key in dribble_idxs:
                    for i, frame in enumerate(action):
                        if len(frame) >= 10: dribbles.append(len(frame))
                if flag: break
            if flag: break

        # plot bar chart for each action
        x_axis_shoot = sorted(set(shoots))
        y_axis_shoot = [shoots.count(i) for i in x_axis_shoot]
        x_axis_pass = sorted(set(passes))
        y_axis_pass = [passes.count(i) for i in x_axis_pass]
        x_axis_dribble = sorted(set(dribbles))
        y_axis_dribble = [dribbles.count(i) for i in x_axis_dribble]
        plt.bar(x_axis_shoot, y_axis_shoot)
        total = Line2D([0], [0], color='none', linestyle='None', label=f'Action total: {len(shoots)}')
        plt.legend(handles=[total], loc='upper right', fontsize=12)
        plt.xlabel('Length of action (frames)')
        plt.ylabel('Number of actions')
        plt.title('Shoot')
        plt.tight_layout()
        plt.show()
        plt.bar(x_axis_pass, y_axis_pass)
        total = Line2D([0], [0], color='none', linestyle='None', label=f'Action total: {len(passes)}')
        plt.legend(handles=[total], loc='upper right', fontsize=12)
        plt.xlabel('Length of action (frames)')
        plt.ylabel('Number of actions')
        plt.title('Pass')
        plt.tight_layout()
        plt.show()
        plt.bar(x_axis_dribble, y_axis_dribble)
        total = Line2D([0], [0], color='none', linestyle='None', label=f'Action total: {len(dribbles)}')
        plt.legend(handles=[total], loc='upper right', fontsize=12)
        plt.xticks(ticks=range(10, max(x_axis_dribble), 10), rotation=45)
        plt.xlabel('Length of action (frames)')
        plt.ylabel('Number of actions')
        plt.title('Dribble')
        plt.tight_layout()
        plt.show()

        print(f"Number of shoots: {len(shoots)}")
        print(f"Number of passes: {len(passes)}")
        print(f"Number of dribbles: {len(dribbles)}")
        print("Average length of shoots: {:.2f}".format(sum(shoots)/len(shoots)))
        print("Average length of passes: {:.2f}".format(sum(passes)/len(passes)))
        print("Average length of dribbles: {:.2f}".format(sum(dribbles)/len(dribbles)))
    else:
        print("The loaded data is not a dictionary.")

# Select function to run

# with open('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_multisports_alltrain.pkl', 'rb') as file:
#     data = pickle.load(file)
# count_classes(data)

compare_class1_correct()
# compare_to_cnnlstm_pkl(data_1, data_2, file_path_1, file_path_2)
# compare_splits(data_1, data_2, file_path_1, file_path_2)
# multisport_pkl(data_2)
# flip_anno_coords(data_1, file_path_1)
# print_split_samples(data_1, file_path_1)
# print_split_samples(data_2, file_path_2)

# for pkl_file in os.listdir('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format'):
#     if pkl_file.endswith('4class.pkl'):
#         file_path = os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format', pkl_file)
#         with open(file_path, 'rb') as file:
#             data = pickle.load(file)
#         print(f"{pkl_file}:")
#         flip_anno_coords(data, file_path)
# convert_4class_to_2class(file_path_2)
# generate_train_val_test(data_1, file_path_1, '4class')
# check_for_duplicate(data_1, file_path_1)
# check_for_duplicate(data_2, file_path_2)
# get_train_weights(data_1, file_path_1)

# with open('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror_normalized.pkl', 'rb') as file:
#     data = pickle.load(file)
# convert_split_to_cnnlstm_format(data, '/media/ogatalab/SSD-PGU3C/Farhan/Doctoral/CNNLSTM/work_dirs/cnntcn_fold_1')