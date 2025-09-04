import os
import pandas as pd

folder_path = '/media/ogatalab/OgataLab8TB/captured_frame/Shoot_HRNet'

# Walk through the directory tree
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.xlsx'):
            print(f'Converting {file} to csv...')
            if 'bboxes' not in file:
                data = pd.read_excel(os.path.join(root, file), header=None)
                data.to_csv(os.path.join(root, os.path.splitext(file)[0] + '.csv'), index=False, header=False)
            else:
                data = pd.read_excel(os.path.join(root, file))
                data.to_csv(os.path.join(root, os.path.splitext(file)[0] + '.csv'), index=False)
            os.remove(os.path.join(root, file))

print('Done!')