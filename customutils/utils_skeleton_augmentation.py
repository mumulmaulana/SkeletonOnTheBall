import numpy as np
import pandas as pd
import argparse
import os

# Define the indices for left and right keypoints in the COCO format
left_right_pairs = [
    (1, 2),  # Left Eye - Right Eye
    (3, 4),  # Left Ear - Right Ear
    (5, 6),  # Left Shoulder - Right Shoulder
    (7, 8),  # Left Elbow - Right Elbow
    (9, 10), # Left Wrist - Right Wrist
    (11, 12), # Left Hip - Right Hip
    (13, 14), # Left Knee - Right Knee
    (15, 16)  # Left Ankle - Right Ankle
]

def mirror_skeleton_sequence(sequence):
    """
    Mirrors a sequence of skeleton frames horizontally.
    
    Parameters:
        sequence (np.ndarray): Array of shape (21, 17, 3), where each frame contains 17 keypoints with (y, x, confidence) coordinates.
        
    Returns:
        np.ndarray: Mirrored sequence of the same shape.
        np.ndarray: (augmentation_type, scaling_factor, x_offset, y_offset) tuple.
    """
    # Copy the sequence to avoid modifying the original
    mirrored_sequence = np.copy(sequence)
    
    # Flip the x-coordinates of each keypoint in each frame
    mirrored_sequence[:, :, 1] = 1280 - mirrored_sequence[:, :, 1]  # negate x-coordinates to the x-length for horizontal flip
    
    # Swap left and right keypoints to keep anatomical correctness
    for left, right in left_right_pairs:
        mirrored_sequence[:, [left, right], :] = mirrored_sequence[:, [right, left], :]
    
    return mirrored_sequence, [['mirror', 1.0, 0, 0] for _ in range(len(sequence))]

def scale_keypoints(sequence):
    """
    Scale a sequence of keypoint coordinates by a scaling factor.

    Parameters:
        keypoints_sequence (np.ndarray): Shape (num_frames, 17, 2), containing keypoints (y, x, confidence).

    Returns:
        np.ndarray: Scaled keypoints sequence.
        np.ndarray: (augmentation_type, scaling_factor, x_offset, y_offset) tuple.
    """
    scaling_factor = np.random.choice(np.concatenate([np.arange(0.5, 1.0, 0.1), np.arange(1.1, 1.6, 0.1)]))  # Random scaling factor between 0.8 and 1.5

    # Calculate the center of the keypoints for scaling
    center_x = np.mean(sequence[:, :, 0])
    center_y = np.mean(sequence[:, :, 1])
    
    # Scale each keypoint in each frame around the center point
    scaled_sequence = np.copy(sequence)
    scaled_sequence[:, :, 1] = (scaled_sequence[:, :, 1] - center_x) * scaling_factor + center_x
    scaled_sequence[:, :, 0] = (scaled_sequence[:, :, 0] - center_y) * scaling_factor + center_y

    # Ensure no negative coordinates by translating by the max negative coordinate
    min_x = scaled_sequence[:, :, 1].min()
    min_y = scaled_sequence[:, :, 0].min()
    if min_x < 0:
        scaled_sequence[:, :, 1] += abs(min_x)+20
    if min_y < 0:
        scaled_sequence[:, :, 0] += abs(min_y)+20
    
    return scaled_sequence, [['scale', scaling_factor, 0, 0] for _ in range(len(sequence))]

def translate_keypoints(sequence):
    """
    Translate a sequence of keypoints by a specified offset.

    Parameters:
        keypoints_sequence (np.ndarray): Shape (num_frames, 17, 3), containing keypoints (y, x, confidence).

    Returns:
        np.ndarray: Translated keypoints sequence with shape (num_frames, 17, 3).
        np.ndarray: (augmentation_type, scaling_factor, x_offset, y_offset) tuple.
    """
    x_offset = np.random.choice(np.concatenate([np.arange(-100, -20, 5), np.arange(20, 100, 5)]))  # Random horizontal shift
    y_offset = np.random.choice(np.concatenate([np.arange(-100, -20, 5), np.arange(20, 100, 5)]))   # Random vertical shift
    
    # Translate the keypoints
    translated_sequence = np.copy(sequence)
    translated_sequence[:, :, 1] += x_offset  # Apply horizontal translation
    translated_sequence[:, :, 0] += y_offset  # Apply vertical translation

    # Ensure no negative coordinates by translating by the max negative coordinate
    min_x = translated_sequence[:, :, 1].min()
    min_y = translated_sequence[:, :, 0].min()
    if min_x < 0:
        translated_sequence[:, :, 1] += abs(min_x)+20
    if min_y < 0:
        translated_sequence[:, :, 0] += abs(min_y)+20
    
    return translated_sequence, [['translate', 1.0, x_offset, y_offset] for _ in range(len(sequence))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to augment skeleton sequences of on-the-ball players from CSV files')
    parser.add_argument('csv_folder', type=str, help='Path to the CSV folder', default='All')
    parser.add_argument('--type', type=str, help='Select the augmentation to perform. Option: [mirror, scale, translate, all]', default='all')
    args = parser.parse_args()
    
    if args.csv_folder == 'All':
        folders = ['/media/ogatalab/OgataLab8TB/captured_frame/Dribble_HRNet', '/media/ogatalab/OgataLab8TB/captured_frame/Pass_HRNet', '/media/ogatalab/OgataLab8TB/captured_frame/Shoot_HRNet']
    else:
        folders = [args.csv_folder]

    np.random.seed(42)
    for csv_folder in folders:
        for root, dirs, files in os.walk(csv_folder):
            for file in files:
                if file.endswith('.csv') and file.split('_')[-1] not in ('bboxes.csv', 'ballbboxes.csv', 'augmented.csv'):
                    skeleton_file = pd.read_csv(os.path.join(root, file), header=None)
                    on_the_ball = skeleton_file[skeleton_file.iloc[:, 2] == 1]
                    if not on_the_ball.empty:
                        interleaved_rows = []
                        print(f'Processing {file}...')
                        if args.type == 'mirror' or args.type == 'all':
                            mirrored_sequence, augmentation_detail = mirror_skeleton_sequence(on_the_ball.iloc[:, 3:].values.reshape(-1, 17, 3))
                            mirrored_sequence = np.concatenate((on_the_ball.iloc[:, :3].values, augmentation_detail, mirrored_sequence.reshape(-1, 51)), axis=1)
                        if args.type == 'scale' or args.type == 'all':
                            scaled_sequence, augmentation_detail = scale_keypoints(on_the_ball.iloc[:, 3:].values.reshape(-1, 17, 3))
                            scaled_sequence = np.concatenate((on_the_ball.iloc[:, :3].values, augmentation_detail, scaled_sequence.reshape(-1, 51)), axis=1)
                        if args.type == 'translate' or args.type == 'all':
                            translated_sequence, augmentation_detail = translate_keypoints(on_the_ball.iloc[:, 3:].values.reshape(-1, 17, 3))
                            translated_sequence = np.concatenate((on_the_ball.iloc[:, :3].values, augmentation_detail, translated_sequence.reshape(-1, 51)), axis=1)

                        # Interleave the sequences
                        for i in range(len(on_the_ball)):
                            if args.type == 'mirror' or args.type == 'all':
                                insert_row = mirrored_sequence[i].copy()
                                insert_row[1] = 90
                                insert_row[2] = 0
                                interleaved_rows.append(insert_row)
                            if args.type == 'scale' or args.type == 'all':
                                insert_row = scaled_sequence[i].copy()
                                insert_row[1] = 91
                                insert_row[2] = 0
                                interleaved_rows.append(insert_row)
                            if args.type == 'translate' or args.type == 'all':
                                insert_row = translated_sequence[i].copy()
                                insert_row[1] = 92
                                insert_row[2] = 0
                                interleaved_rows.append(insert_row)
                        augmented_sequence = pd.DataFrame(interleaved_rows)
                        augmented_sequence.to_csv(os.path.join(root, file.replace('.csv', '_augmented.csv')), index=False, header=False)
