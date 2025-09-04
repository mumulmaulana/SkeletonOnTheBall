import os
import json
import argparse
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter

# Function to initialize a Kalman filter for a single point
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])
    kf.P *= 1000
    kf.R = np.array([[5, 0],
                     [0, 5]])
    kf.Q = np.eye(4)
    return kf

def preprocess_kalman(skeleton_file, action):
    df = pd.read_csv(os.path.join('/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action), skeleton_file), header=None)
    df = df.rename(columns={df.columns[0]: 'frame', df.columns[1]: 'player_id', df.columns[2]: 'in_action'})

    # get all rows where in_action is 1
    in_action_df = df[df['in_action'] == 1]
    kalman_filters = [create_kalman_filter() for _ in range(17)]
    filtered_keypoints = []

    # apply Kalman filter to in_action rows
    kf = create_kalman_filter()
    for frame_num, row in in_action_df.iterrows():
        for i in range(17):
            x, y, confidence = row[3 + i*3], row[4 + i*3], row[5 + i*3]
            if confidence > 0:  # Only update if confidence is greater than 0
                z = np.array([x, y])
                kalman_filters[i].update(z)
                kalman_filters[i].predict()

                filtered_keypoints[frame_num].append(np.append(kalman_filters[i].x[:2], 0.8))

    # create new dataframe with filtered keypoints by matching frame number
    for frame_num, keypoint in enumerate(filtered_keypoints):
        df.loc[df['frame'] == frame_num and df['in_action'] == 1, 3:] = keypoint[1:]

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to extract skeleton files list from a .txt file and preprocess skeleton files from the from the list')
    parser.add_argument('txt_file', type=str, help='Path to the JSON file', default='./skeleton_files_with_skips_Dribble.txt')
    args = parser.parse_args()

    txt_file = args.txt_file
    action = txt_file.split('_')[4].split('.')[0]
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), txt_file)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocess_output')
    with open(file_path, 'r') as file:
        skeleton_files = file.readlines()
        for file in skeleton_files:
            preprocessed_csv = preprocess_kalman(file.strip() + '.csv', action)
            preprocessed_csv.to_csv(os.path.join(save_path, file.strip() + '_preprocessed.csv'), index=False)