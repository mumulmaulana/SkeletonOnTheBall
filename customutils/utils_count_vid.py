import os
import json

folder_path = '/media/ogatalab/OgataLab8TB/captured_frame/Shoot_HRNet'
filename_list = []

# Walk through the directory tree
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.mp4'):
            filename_list.append(os.path.join(root, file))
    
# Initialize dictionaries to count videos
vid_counts = {}

for filename in filename_list:
    # split the filename and count the video
    file = filename.split('/')[-1]
    vid = file.split('_')[0][:-1]
    if vid in vid_counts:
        vid_counts[vid] += 1
    else:
        vid_counts[vid] = 1

# Print the count of each label (optional)
for label, count in vid_counts.items():
    print(f"{label}: {count}")