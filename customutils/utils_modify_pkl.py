import pickle
import re
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

np.random.seed(0)

def add_augmentation():
    aug_name = 'pkl_format_fold_1_new_all' 
    aug_path = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/'+aug_name+'.pkl'
    with open(aug_path, 'rb') as file:
        aug_data = pickle.load(file)
    annos = aug_data['annotations']
    annos_indexed = {anno['frame_dir']: anno for anno in annos}

    split_types = ['mirror', 'scale', 'translate', 'all', 'random']
    # split_types = ['random']
    classes_dict = {split_type: [[] for j in range(10)] for split_type in split_types}

    save_dir = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mode = '2class'
    # found_idx = 0 #CAN DELETE IF ALL ELEMENTS ARE FOUND
    for i in range(1,11):
        new_split = {split_type: {'split': {'train': [], 'val': [], 'test': []}, 'annotations': []} for split_type in split_types}

        pkl_name = 'pkl_format_fold_{}_new_{}'.format(i, mode)
        pkl_path = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/'+pkl_name+'.pkl'
        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)

        print('Processing {}'.format(pkl_name))
        split = pkl_data['split']
        anns = pkl_data['annotations']
        # print('Anns[0] key: {}'.format(next(iter(anns)))) #CAN DELETE IF ALL ELEMENTS ARE FOUND
        index_table = {ann['frame_dir']: idx for idx, ann in enumerate(anns)}
        for key, value in split.items():
            print('{} split'.format(key))
            total = len(value)
            with tqdm(total=total, desc="Starting") as pbar:
                for iter, element in enumerate(value):
                    get_result = annos_indexed.get(element, 0)
                    new_element = element # in case of no match, copy the element for future check
                    if get_result == 0:
                        # example = '2015-02-21-17-30Paderborn0-6BayernMunich1_720p_55254_Shoot_on'
                        pattern = r"_(Shoot|Pass|Dribble)_"
                        new_element = re.sub(pattern, '_', new_element)
                        get_result = annos_indexed.get(new_element, 0)
                    
                    pbar.update(1)

                    if get_result != 0:
                        if new_element != element:
                            get_result['frame_dir'] = element
                        # update label for the new annos
                        get_result['label'] = anns[index_table[element]]['label']

                        for split_type in split_types:
                            pbar.set_description(f"[{iter+1}/{total}] ({split_type}) Processing {element}")

                            if int(get_result['label']) < 1 or split_type != 'random': # if split_type is random, only add the data if the label is 0
                                new_split[split_type]['split'][key].append(element)
                                new_split[split_type]['annotations'].append(get_result)
                                if key == 'train': classes_dict[split_type][i-1].append(int(get_result['label']))

                            if int(get_result['label']) >= 1:
                                get_result_mirror = annos_indexed.get(new_element+'_mirror', 0)
                                get_result_scale = annos_indexed.get(new_element+'_scale', 0)
                                get_result_translate = annos_indexed.get(new_element+'_translate', 0)
                                random_choice = np.random.choice(['mirror', 'scale', 'translate', 'original'])
                                if random_choice == 'original':
                                    get_result_random = get_result
                                else:
                                    get_result_random = annos_indexed.get(new_element+'_'+random_choice, 0)
                                # update label for the new annos
                                get_result_mirror['label'] = anns[index_table[element]]['label']
                                get_result_scale['label'] = anns[index_table[element]]['label']
                                get_result_translate['label'] = anns[index_table[element]]['label']
                                get_result_random['label'] = anns[index_table[element]]['label']
                                
                                if split_type == 'mirror':
                                    get_result_mirror['frame_dir'] = element+'_mirror'
                                    new_split[split_type]['split'][key].append(element+'_mirror')
                                    new_split[split_type]['annotations'].append(get_result_mirror)
                                    if key == 'train': classes_dict[split_type][i-1].append(int(get_result_mirror['label']))
                                elif split_type == 'scale':
                                    get_result_scale['frame_dir'] = element+'_scale'
                                    new_split[split_type]['split'][key].append(element+'_scale')
                                    new_split[split_type]['annotations'].append(get_result_scale)
                                    if key == 'train': classes_dict[split_type][i-1].append(int(get_result_scale['label']))
                                elif split_type == 'translate':
                                    get_result_translate['frame_dir'] = element+'_translate'
                                    new_split[split_type]['split'][key].append(element+'_translate')
                                    new_split[split_type]['annotations'].append(get_result_translate)
                                    if key == 'train': classes_dict[split_type][i-1].append(int(get_result_translate['label']))
                                elif split_type == 'all':
                                    get_result_mirror['frame_dir'] = element+'_mirror'
                                    get_result_scale['frame_dir'] = element+'_scale'
                                    get_result_translate['frame_dir'] = element+'_translate'
                                    new_split[split_type]['split'][key].append(element+'_mirror')
                                    new_split[split_type]['split'][key].append(element+'_scale')
                                    new_split[split_type]['split'][key].append(element+'_translate')
                                    new_split[split_type]['annotations'].append(get_result_mirror)
                                    new_split[split_type]['annotations'].append(get_result_scale)
                                    new_split[split_type]['annotations'].append(get_result_translate)
                                    if key == 'train':
                                        classes_dict[split_type][i-1].append(int(get_result_mirror['label']))
                                        classes_dict[split_type][i-1].append(int(get_result_scale['label']))
                                        classes_dict[split_type][i-1].append(int(get_result_translate['label']))
                                elif split_type == 'random':
                                    save_element = element if random_choice == 'original' else element+'_'+random_choice
                                    get_result_random['frame_dir'] = save_element
                                    new_split[split_type]['split'][key].append(save_element)
                                    new_split[split_type]['annotations'].append(get_result_random)
                                    if key == 'train': classes_dict[split_type][i-1].append(int(get_result_random['label']))
                    
                    else:
                        raise ValueError(f"Element {element} not found in the annotations")

        print("Dumping new pkl format for {}".format(pkl_name))
        for key, splits in new_split.items():
            pkl = dict()
            pkl['split'] = splits['split']
            pkl['annotations'] = splits['annotations']

            pkl_name_save = '_'.join(pkl_name.split('_')[:-1])
            with open(os.path.join(save_dir, '{}_{}_{}.pkl'.format(pkl_name_save, mode, key)), 'wb') as f:
                pickle.dump(pkl, f)

    for key, classes in classes_dict.items():
        class_weights_list = []
        for i, class_list in enumerate(classes):
            # Compute class weights for the training data
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_list), y=class_list)
            print(f"Class weights for {key} Fold {i + 1}: {class_weights}")
            class_weights_list.append(class_weights)

        with open(os.path.join(save_dir, 'class_weights_{}_{}.txt'.format(mode, key)), 'w') as f:
            for i, class_weight in enumerate(class_weights_list):
                f.write(f"Fold {i + 1}: {class_weight}\n")

def reduce_off_class(data, file_path):
    if isinstance(data, dict):
        annos = data['annotations']
        annos_indexed = {anno['frame_dir']: anno for anno in annos}
        split_combined = data['split']['train'] + data['split']['val'] + data['split']['test']
        new_split = {'train': [], 'val': [], 'test': []}
        new_annos = []
        random_pick = 5 # number of off class to pick
        buffer_pick = 0
        for split in ['train', 'val', 'test']:
            total = len(data['split'][split])
            with tqdm(total=total, desc="Finding...") as pbar:
                for i, element in enumerate(data['split'][split]):
                    pbar.update(1)
                    get_result = annos_indexed.get(element, 0)
                    if get_result['label'] == 0:
                        continue
                    else:
                        pbar.set_description(f"({i+1}/{total}) Processing {element}")
                        new_split[split].append(element)
                        new_annos.append(get_result)
                        # find all off class if element is not mirror
                        if '_mirror' not in element:
                            video_name = element.split('_on')[0]
                            off_elements = np.array([off_element for off_element in split_combined if video_name in off_element and '_off' in off_element])
                            # raise ValueError(video_name, len(off_elements))
                            current_pick = random_pick + buffer_pick
                            if len(off_elements) < current_pick:
                                current_pick = len(off_elements)
                                buffer_pick += random_pick - current_pick
                                # raise ValueError(video_name, len(off_elements))
                            else:
                                buffer_pick = 0
                            random_pick_off = off_elements[np.random.choice(len(off_elements), current_pick, replace=False)]
                            for off in random_pick_off:
                                get_result_off = annos_indexed.get(off, 0)
                                if get_result_off != 0:
                                    new_split[split].append(off)
                                    new_annos.append(get_result_off)

        new_data = {'split': new_split, 'annotations': new_annos}
        save_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name = file_name+'_'+str(random_pick)+'off.pkl'
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(new_data, f)
    else:
        print("The loaded data is not a dictionary.")

if __name__ == "__main__":
    # add_augmentation()

    file_path_1 = '/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror_normalized.pkl'
    with open(file_path_1, 'rb') as file_1:
        data_1 = pickle.load(file_1)
    reduce_off_class(data_1, file_path_1)