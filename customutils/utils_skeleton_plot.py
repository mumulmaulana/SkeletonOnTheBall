import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm

# COCO keypoint names
COCO_KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 
                  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
                  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
# COCO keypoint connections
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # Nose to Eyes and Ears
    # (0, 5), (0, 6),                   # Nose to Shoulders
    (5, 6),                           # Shoulders
    (5, 7), (6, 8),                   # Shoulders to Elbows
    (7, 9), (8, 10),                  # Elbows to Wrists
    (5, 11), (6, 12),                 # Shoulders to Hips
    (11, 12),                         # Hips
    (11, 13), (12, 14),               # Hips to Knees
    (13, 15), (14, 16)                # Knees to Ankles
]

def pkl_load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    res = dict()
    res['train'] = data['split']['train']
    res['val'] = data['split']['val']
    res['test'] = data['split']['test']
    res['annotations'] = data['annotations']
    return res

def get_split_annos(split, annos):
    annos_indexed = {anno['frame_dir']: anno for anno in annos}
    split_annos = [annos_indexed[frame_dir] for frame_dir in split]
    return split_annos

def normalize_skeletons(split_annos, frame_shape=(1280, 720), target_size=200, recenter=True, resize=True):
    skeletons = np.array([anno['keypoint'] for anno in split_annos])  # Shape: (N, 1, T, V, 2)

    # Step 1: Remove the singleton dimension
    skeletons = skeletons.squeeze(1)  # New shape: (N, T, V, 2)

    if recenter:
        frame_center = np.array([frame_shape[0] / 2, frame_shape[1] / 2])  # (center_x, center_y)

        # Step 2: Calculate the midpoint of left hip (index 11) and right hip (index 12)
        left_hip = skeletons[:, :, COCO_KEYPOINTS.index('left_hip'), :]  # Shape: (N, T, 2)
        right_hip = skeletons[:, :, COCO_KEYPOINTS.index('right_hip'), :]  # Shape: (N, T, 2)
        hip_midpoint = (left_hip + right_hip) / 2  # Shape: (N, T, 2)
        offsets = frame_center - hip_midpoint  # Shape: (N, T, 2)

        # Step 3: Center all skeletons
        offsets_expanded = offsets[:, :, np.newaxis, :]  # Shape: (N, T, 1, 2)
        centered_skeletons = skeletons + offsets_expanded  # Shape: (N, T, V, 2)
    else:
        centered_skeletons = skeletons

    if resize:
        # Step 4: Calculate current size (bounding box size) for each frame
        min_coords = np.min(centered_skeletons, axis=2)  # Shape: (N, T, 2)
        max_coords = np.max(centered_skeletons, axis=2)  # Shape: (N, T, 2)
        current_sizes = np.linalg.norm(max_coords - min_coords, axis=-1)  # Shape: (N, T)

        # Step 5: Compute per-frame scaling factors and apply to each frame
        # scaling_factors = target_size / current_sizes  # Shape: (N, T)
        # scaled_skeletons = centered_skeletons * scaling_factors[:, :, np.newaxis, np.newaxis]

        # # Step 5: Compute average size per video and apply scaling to each video
        avg_sizes = np.mean(current_sizes, axis=1)  # Shape: (N,)
        scaling_factors = target_size / avg_sizes  # Shape: (N,)
        scaled_skeletons = centered_skeletons * scaling_factors[:, np.newaxis, np.newaxis, np.newaxis]

        # Step 6: Re-center skeletons to ensure they stay in the frame center (if recenter=True)
        if recenter:
            final_offsets = frame_center - np.mean(scaled_skeletons[:, :, [COCO_KEYPOINTS.index('left_hip'), COCO_KEYPOINTS.index('right_hip')], :].mean(axis=2), axis=1)
            final_offsets = final_offsets[:, np.newaxis, np.newaxis, :]  # Shape: (N, 1, 1, 2)
        else:
            final_offsets = np.zeros((len(scaled_skeletons), 1, 1, 2))

        normalized_skeletons = scaled_skeletons + final_offsets  # Shape: (N, T, V, 2)
    else:
        normalized_skeletons = centered_skeletons

    # Step 7:  Restore the original dimension (N, 1, T, V, 2) and replace split annos with normalized skeletons
    normalized_skeletons = np.expand_dims(normalized_skeletons, axis=1)
    # print(f"Normalized skeletons shape: {normalized_skeletons.shape}")
    for i, anno in enumerate(split_annos):
        anno['keypoint'] = normalized_skeletons[i]

    return split_annos

def get_bbox_avgsize(split_annos, h):
    keypoints = np.array([anno['keypoint'][0] for anno in split_annos])  # Shape: (N, T, V, 2)

    # Get y-coordinates of hip midpoints
    left_hip = keypoints[:, :, COCO_KEYPOINTS.index('left_hip'), :]  # Shape: (N, T, 2)
    right_hip = keypoints[:, :, COCO_KEYPOINTS.index('right_hip'), :]  # Shape: (N, T, 2)
    hip_midpoint = (left_hip + right_hip) / 2  # Shape: (N, T, 2)
    hip_midpoint_y = hip_midpoint[:, :, 1]  # Shape: (N, T)
    # separate bounding boxes according to threshold
    threshold = h // 2

    # Create masks for upper and lower halves
    upper_mask = hip_midpoint_y < threshold  # True for hip midpoint y < separator
    lower_mask = hip_midpoint_y >= threshold  # True for hip midpoint y >= separator
    # Expand masks to match keypoints shape
    upper_mask = upper_mask[:, :, np.newaxis, np.newaxis]  # Shape: (N, T, 1, 1)
    lower_mask = lower_mask[:, :, np.newaxis, np.newaxis]  # Shape: (N, T, 1, 1)
    # Separate keypoints based on masks
    upper_half_keypoints = np.where(upper_mask, keypoints, np.nan)  # Set non-matching to NaN
    lower_half_keypoints = np.where(lower_mask, keypoints, np.nan)  # Set non-matching to NaN

    print(f"Upper half keypoints shape: {upper_half_keypoints.shape}")
    print(f"Lower half keypoints shape: {lower_half_keypoints.shape}")

def plot_xy_intensity(split_annos, w, h):
    keypoints = np.array([anno['keypoint'][0] for anno in split_annos])
    labels = np.array([anno['label'] for anno in split_annos])

    # Step 2: Calculate the midpoint of left hip (index 11) and right hip (index 12)
    left_hip = keypoints[:, :, COCO_KEYPOINTS.index('left_hip'), :]  # Shape: (N, T, 2)
    right_hip = keypoints[:, :, COCO_KEYPOINTS.index('right_hip'), :]  # Shape: (N, T, 2)
    hip_midpoint = (left_hip + right_hip) / 2  # Shape: (N, T, 2)

    # Flatten the keypoints array and filter out invalid coordinates
    x_coords = hip_midpoint[:, :, 0].flatten()
    y_coords = hip_midpoint[:, :, 1].flatten()
    valid_indices = (y_coords >= 0) & (y_coords < h) & (x_coords >= 0) & (x_coords < w)
    x_coords = x_coords[valid_indices].astype(int)
    y_coords = y_coords[valid_indices].astype(int)
    labels = np.repeat(labels, hip_midpoint.shape[1])[valid_indices]

    # Update heatmaps
    heatmap_class0 = np.zeros((w, h), dtype=np.uint8)
    heatmap_class1 = np.zeros((w, h), dtype=np.uint8)
    np.add.at(heatmap_class0, (x_coords[labels == 0], y_coords[labels == 0]), 1)
    np.add.at(heatmap_class1, (x_coords[labels == 1], y_coords[labels == 1]), 1)

    # Find bounding box to crop the heatmaps so that it only contains the valid coordinates
    # x_min, x_max = np.min(x_coords), np.max(x_coords)
    # y_min, y_max = np.min(y_coords), np.max(y_coords)
    # print(f'Bounding box resolution: {x_max - x_min} x {y_max - y_min}')
    # heatmap_class0 = heatmap_class0[(x_min-5):(x_max+5), (y_min-5):(y_max+5)]
    # heatmap_class1 = heatmap_class1[(x_min-5):(x_max+5), (y_min-5):(y_max+5)]

    # Normalize the heatmap so that the values are between 0 and 255
    # heatmap_class0 = heatmap_class0 / np.max(heatmap_class0) * 255
    # heatmap_class1 = heatmap_class1 / np.max(heatmap_class1) * 255

    # Plot the heatmaps
    _, ax = plt.subplots(2, 1, figsize=(6, 6))
    sns.heatmap(heatmap_class0.T, ax=ax[0], cmap='gray_r', cbar_kws={'label': 'Player count'})
    sns.heatmap(heatmap_class1.T, ax=ax[1], cmap='gray_r', cbar_kws={'label': 'Player count'})
    ax[0].set_title('Off the ball heatmap')
    ax[1].set_title('On the ball heatmap')
    for a in ax:
        cbar = a.collections[0].colorbar
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

def plot_movement(split_annos):
    keypoints = np.array([anno['keypoint'][0] for anno in split_annos])  # Shape: (N, T, V, 2)
    labels = np.array([anno['label'] for anno in split_annos])
    
    # Displacement calculations by labels
    all_displacements = np.linalg.norm(np.diff(keypoints, axis=1), axis=3)  # Shape: (N, T-1, V)
    unique_labels = np.unique(labels)
    grouped_displacements = {label: all_displacements[labels == label] for label in unique_labels}

    # AVERAGE CUMULATIVE MOVEMENT (All joints)
    grouped_cumulative_displacement = {
        label: np.cumsum(group, axis=1) for label, group in grouped_displacements.items()
    }

    # displacement ratio between each joint
    for label, group in grouped_displacements.items():
        avg_displacement = np.mean(group, axis=0)
        min_displacement = np.min(avg_displacement)
        max_displacement = np.max(avg_displacement)
        # biases = 1 + ((avg_displacement - min_displacement) / (max_displacement - min_displacement) - 0.5) * 2
        # print(biases.shape)
        # print(f"Normalize displacement for class {label}: {biases[-1]}")

    global_max = max(
        np.max(cumulative_displacement.mean(axis=0))
        for cumulative_displacement in grouped_cumulative_displacement.values() # Global maximum value across all groups for consistent colorbar scaling
    )
    # Adjust subplot layout based on the number of unique labels
    if len(unique_labels) == 2:
        fig, axes = plt.subplots(len(unique_labels), 1, figsize=(6, 6), sharey=True)
        axes_iter = axes
        label_names = ["Off the ball", "On the ball"]
    else:
        subplot_axis = unique_labels.reshape(-1, 2)
        fig, axes = plt.subplots(len(subplot_axis[0]), len(subplot_axis[1]), figsize=(12, 6), sharey=True)
        axes_iter = axes.reshape(-1)
        label_names = ["Off the ball", "Pass", "Dribble", "Shoot"]
    # Subplots for heatmaps
    for ax, label, name in zip(axes_iter, unique_labels, label_names):
        cumulative_displacement = grouped_cumulative_displacement[label]
        average_cumulative_displacement = cumulative_displacement.mean(axis=0)  # Average across videos
        # Plot heatmap
        im = ax.imshow(
            average_cumulative_displacement.T, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=global_max
        )
        ax.set_title(f"{name}")
        ax.set_xlabel("Time ($Frame_{t+1}$ - $Frame_{t}$)")
        ax.set_xticks(range(0, average_cumulative_displacement.shape[0], 2))
        ax.set_xticklabels(range(1, average_cumulative_displacement.shape[0]+1, 2))
        ax.set_ylabel("Joint")
        ax.set_yticks(np.arange(len(COCO_KEYPOINTS)))
        ax.set_yticklabels(COCO_KEYPOINTS)
        fig.colorbar(im, ax=ax, orientation="vertical", label="Avg Cumulative Motion")
    plt.tight_layout()
    plt.show()

    # AVERAGE DISPLACEMENT (Per joint)
    grouped_avg_displacement = {
        label: np.mean(group, axis=0) for label, group in grouped_displacements.items() 
    }
    global_max = max(
        np.max(avg_displacement)
        for avg_displacement in grouped_avg_displacement.values() # Global maximum value across all groups for consistent colorbar scaling
    )
    # Adjust subplot layout based on the number of unique labels
    if len(unique_labels) == 2:
        fig, axes = plt.subplots(len(unique_labels), 1, figsize=(6, 6), sharey=True)
        axes_iter = axes
        label_names = ["Off the ball", "On the ball"]
    else:
        subplot_axis = unique_labels.reshape(-1, 2)
        fig, axes = plt.subplots(len(subplot_axis[0]), len(subplot_axis[1]), figsize=(12, 6), sharey=True)
        axes_iter = axes.reshape(-1)
        label_names = ["Off the ball", "Pass", "Dribble", "Shoot"]
    # Subplots for heatmaps
    for ax, label, name in zip(axes_iter, unique_labels, label_names):
        avg_displacement = grouped_avg_displacement[label]
        # Plot heatmap
        im = ax.imshow(
            avg_displacement.T, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=global_max
        )
        ax.set_title(f"{name}")
        # ax.set_xlabel("Time ($Frame_{t+1}$ - $Frame_{t}$)")
        ax.set_xlabel("Frame")
        ax.set_xticks(range(0, avg_displacement.shape[0], 2))
        ax.set_xticklabels(range(1, avg_displacement.shape[0]+1, 2))
        ax.set_ylabel("Joint")
        ax.set_yticks(np.arange(len(COCO_KEYPOINTS)))
        ax.set_yticklabels(COCO_KEYPOINTS)
        fig.colorbar(im, ax=ax, orientation="vertical", label="Avg Joint Motion")
    plt.tight_layout()
    plt.show()

    # Line plot for average displacement per joint
    if len(unique_labels) == 2:
        fig, axes = plt.subplots(len(unique_labels), 1, figsize=(6, 8), sharey=True)
        axes_iter = axes
        label_names = ["Off the ball", "On the ball"]
    else:
        subplot_axis = unique_labels.reshape(-1, 2)
        fig, axes = plt.subplots(len(subplot_axis[0]), len(subplot_axis[1]), figsize=(12, 8), sharey=True)
        axes_iter = axes.reshape(-1)
        label_names = ["Off the ball", "Pass", "Dribble", "Shoot"]
    for ax, label, name in zip(axes_iter, unique_labels, label_names):
        avg_displacement = grouped_avg_displacement[label]
        ax.plot(avg_displacement, label=COCO_KEYPOINTS)
        ax.set_title(f"{name}")
        ax.set_xlabel("Time ($Frame_{t+1}$ - $Frame_{t}$)")
        ax.set_xticks(range(0, avg_displacement.shape[0], 2))
        ax.set_xticklabels(range(1, avg_displacement.shape[0]+1, 2))
        ax.set_ylabel("Avg Joint Displacement")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_skeleton(keypoints, connections, ax, title="Skeleton"):
    # Plot skeleton for t
    for x, y in keypoints[0]:
        ax.scatter(x, y, color="red", s=40, alpha=0.3)
    for start, end in connections:
        x_coords = [keypoints[0][start, 0], keypoints[0][end, 0]]
        y_coords = [keypoints[0][start, 1], keypoints[0][end, 1]]
        ax.plot(x_coords, y_coords, color="blue", linewidth=2, alpha=0.3)

    # Plot skeleton for t+1
    for x, y in keypoints[1]:
        ax.scatter(x, y, color="red", s=40)
    for start, end in connections:
        x_coords = [keypoints[1][start, 0], keypoints[1][end, 0]]
        y_coords = [keypoints[1][start, 1], keypoints[1][end, 1]]
        ax.plot(x_coords, y_coords, color="blue", linewidth=2)
    
    # Set plot properties
    ax.set_title(title)
    ax.invert_yaxis()  # Invert Y-axis for image-like coordinates
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', adjustable='datalim')

if __name__ == '__main__':
    data = pkl_load('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_4class_mirror.pkl')
    # splits_key = [key for key in data.keys() if key not in ['annotations']]
    # split_annos = get_split_annos(data['test'], data['annotations'])
    # check_oob(split_annos)

    # Get 2016-09-24-14-30ManchesterUnited4-1Leicester2_720p_30977_Shoot_on from split_annos
    # annos_indexed = {anno['frame_dir']: anno for anno in data['annotations']}
    # target = annos_indexed.get('2016-09-24-14-30ManchesterUnited4-1Leicester2_720p_30977_Shoot_on')
    # for i in range(0, len(target['keypoint'][0])-1):
    #     skeleton = target['keypoint'][0][i:i+2]
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     plot_skeleton(skeleton, COCO_CONNECTIONS, ax)
    #     plt.show()

    # recenter=True
    # resize=True
    # for split in splits_key:
    #     print(f"Plotting {split} split ({len(data[split])} skeletons)")
    #     split_annos = get_split_annos(data[split], data['annotations'])
    #     recentered_annos = normalize_skeletons(split_annos, (1280, 720), recenter=recenter, resize=resize)

        # skeleton = recentered_annos[0]['keypoint'][0][0]
        # fig, ax = plt.subplots(figsize=(6, 6))
        # plot_skeleton(skeleton, COCO_CONNECTIONS, ax)
        # plt.show()

        # plot_xy_intensity(recentered_annos, 1280, 720)
        # plot_movement(recentered_annos)

    # plot_xy_intensity(data['annotations'], 1280, 720)
    # get_bbox_avgsize(data['annotations'], 720)
    recentered_annos = normalize_skeletons(data['annotations'], (1280, 720))
    plot_movement(recentered_annos)
    
    # recenter=False
    # resize=True
    # normalized_annos = normalize_skeletons(data['annotations'], (1280, 720), recenter=recenter, resize=resize)
    # normalized = dict()
    # split = {'train': [], 'val': [], 'test': []}
    # split['train'] = data['train']
    # split['val'] = data['val']
    # split['test'] = data['test']
    # normalized['split'] = split
    # normalized['annotations'] = normalized_annos
    # if recenter and resize:
    #     with open('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror_normalized.pkl', 'wb') as f:
    #         pickle.dump(normalized, f)
    # elif resize:
    #     with open('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror_avgresized.pkl', 'wb') as f:
    #         pickle.dump(normalized, f)
    # else:
    #     with open('/media/ogatalab/OgataLab8TB/captured_frame/new_pkl_format/pkl_format_fold_1_new_2class_mirror_recentered.pkl', 'wb') as f:
    #         pickle.dump(normalized, f)
