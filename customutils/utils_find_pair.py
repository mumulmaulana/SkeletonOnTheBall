import os

def find_unpaired_videos(shoot_folder, shoot_hrnet_folder):
    # List all files in both directories
    shoot_files = [os.path.splitext(file)[0] for file in os.listdir(shoot_folder) if file.endswith('.mp4')]
    shoot_hrnet_files = ['_'.join(os.path.splitext(file)[0].split('_')[:-1]) for file in os.listdir(shoot_hrnet_folder) if file.endswith('.mp4')]

    # Find unpaired files
    unpaired_in_shoot = set(shoot_files) - set(shoot_hrnet_files)
    unpaired_in_shoot_hrnet = set(shoot_hrnet_files) - set(shoot_files)

    return unpaired_in_shoot, unpaired_in_shoot_hrnet

# Define the paths to the folders
action = 'Dribble'
shoot_folder = '/media/ogatalab/OgataLab8TB/captured_frame/{}'.format(action)
shoot_hrnet_folder = '/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action)

# Find unpaired videos
unpaired_in_shoot, unpaired_in_shoot_hrnet = find_unpaired_videos(shoot_folder, shoot_hrnet_folder)

print(f"Unpaired in {action} folder: {unpaired_in_shoot}")
print(f"Unpaired in {action}_HRNet folder: {unpaired_in_shoot_hrnet}")