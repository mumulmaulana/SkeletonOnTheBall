import os
import json
import argparse
import random
import pandas as pd
import numpy as np

def get_skeleton_files_with_skips(json_file):
    skeleton_files = []
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), json_file)
    print(file_path)
    
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        print("Total files: ", len(data['files']))
        for filename in data['files']:
            if (data['files'][filename]['skip_counts_15_35']['one_skip'] > 0 or data['files'][filename]['skip_counts_15_35']['two_skip'] > 0) and data['files'][filename]['skip_counts_15_35']['more_than_two_skip'] == 0:
                skeleton_files.append(os.path.splitext(filename)[0])
    
    return skeleton_files

def get_in_range_skeletons(json_file, length):
    if length == 3: length_str = "24_26"
    elif length == 7: length_str = "22_28"
    elif length == 11: length_str = "20_30"
    elif length == 21: length_str = "15_35"
    else: length_str = "15_35"
    random.seed(42)
    deficit = 0
    
    if json_file == 'All':
        input_files = ["Shoot_action_in_range.json", "Dribble_action_in_range.json", "Pass_action_in_range.json"]
    else:
        input_files = [json_file]

    on_the_ball = []
    off_the_ball = []
    for inp in input_files:
        action = inp.split('_')[0]
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), inp)
        print(file_path)
        # Open and read the JSON file
        with open(file_path, 'r') as file_to_read:
            data = json.load(file_to_read)

            for filename in data['files']:
                # skeleton_file = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), os.path.splitext(filename)[0]+'.csv'), header=None)
                # Append the on_the_ball skeletons to the dictionary
                if len(data['files'][filename]["missing_frames_{}".format(length_str)]) == 0:
                    on_the_ball.append(os.path.splitext(filename)[0] + '_{}'.format(action))

                    # Append the augmented skeletons to the dictionary
                    # for augmentation in ['mirror', 'scale', 'translate']:
                    for augmentation in ['mirror']:
                        on_the_ball.append(os.path.splitext(filename)[0] + '_{}_{}'.format(action, augmentation))

                    # Append the off_the_ball skeletons to the dictionary (new version, use this to get 5 off the ball skeletons)
                    # if len(data['files'][filename]["no_action_ids_{}".format(length_str)]) > 0:
                    #     no_action_ids = data['files'][filename]["no_action_ids_{}".format(length_str)]
                    #     # Select 5 elements randomly from no_action_ids with reproducible seed
                    #     selected_ids = random.sample(no_action_ids, min(deficit+5, len(no_action_ids)))
                    #     # If the number of selected ids is less than 5, add the difference to the deficit
                    #     if deficit + 5 > len(selected_ids):
                    #         deficit = deficit + 5 - len(selected_ids)
                    #     else:
                    #         deficit = 0
                        
                    #     for id in selected_ids: 
                    #         off_the_ball.append(os.path.splitext(filename)[0] + '_{}_off_{}'.format(action, id))

                # Append the off_the_ball skeletons to the dictionary (old version, use this to get all off the ball skeletons)
                if len(data['files'][filename]["no_action_ids_{}".format(length_str)]) > 0:
                    for id in data['files'][filename]["no_action_ids_{}".format(length_str)]: 
                        off_the_ball.append(os.path.splitext(filename)[0] + '_{}_off_{}'.format(action, id))

    return on_the_ball, off_the_ball

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to read JSON files and extract skeleton files that satisfy a condition')
    parser.add_argument('json_file', type=str, help='Path to the JSON file', default='./Shoot_action_in_range.json')
    parser.add_argument('--condition', type=str, help='Select the condition to extract. Option: [skips, range_(length)]', default='skips')
    args = parser.parse_args()

    json_file = args.json_file
    action = json_file.split('_')[0]
    if args.condition == 'skips':
        skeleton_files_with_skips = get_skeleton_files_with_skips(json_file)
        with open('skeleton_files_with_skips_{}.txt'.format(action), 'w') as file:
            for skeleton_file in skeleton_files_with_skips:
                file.write(skeleton_file + '\n')
        print("Finished writing to skeleton_files_with_skips_{}.txt".format(action))
    elif args.condition.startswith('range_'):
        length = int(args.condition.split('_')[1])
        on_the_ball, off_the_ball = get_in_range_skeletons(json_file, length)
        with open('on_the_ball_{}_mirror.json'.format(str(length)), 'w') as file:
            json.dump(on_the_ball, file)
        # with open('off_the_ball_{}.json'.format(str(length)), 'w') as file:
        #     json.dump(off_the_ball, file)
        print("Finished extracting skeleton ids in range {} from {}".format(length, json_file))