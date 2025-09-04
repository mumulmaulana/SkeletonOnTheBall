import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from itertools import groupby
from operator import itemgetter
from matplotlib.lines import Line2D
from collections import Counter

def count_skips(sequence):
    one_skip = 0
    two_skip = 0
    more_than_two_skip = 0

    for i in range(1, len(sequence)):
        diff = sequence[i] - sequence[i - 1]
        if diff == 2:
            one_skip += 1
        elif diff == 3:
            two_skip += 1
        elif diff > 3:
            more_than_two_skip += 1

    return one_skip, two_skip, more_than_two_skip

action = 'All'  # 'Dribble', 'Shoot', 'Pass', 'All'
skip_plottings = False

### IN-ACTIONS PLAYER PER FRAME ###
combined_frame_action_freq = pd.Series(dtype=int)
longest_sequence_dict = {}
filtered_sequences = {}
in_action_frames = {}

if action != 'All':
    action_folder = '/media/ogatalab/OgataLab8TB/captured_frame/{}_HRNet'.format(action)
    skeleton_files = [file for file in os.listdir(action_folder) if file.endswith('.csv') and 'bboxes' not in file and 'ballbboxes' not in file]
    if os.path.exists('{}_action_frequencies.json'.format(action)):
        with open('{}_action_frequencies.json'.format(action), 'r') as json_file:
            combined_frame_action_freq = pd.Series(json.load(json_file))
        with open('{}_action_sequences.json'.format(action), 'r') as json_file:
            longest_sequence_dict = json.load(json_file)
        with open('{}_sequences_in_range.json'.format(action), 'r') as json_file:
            filtered_sequences = json.load(json_file)
        with open('{}_action_in_range.json'.format(action), 'r') as json_file:
            in_action_frames = json.load(json_file)
    else:
        for skeleton_file in skeleton_files:
            print(skeleton_file)
            df = pd.read_excel(os.path.join(action_folder, skeleton_file), header=None)
            df = df.rename(columns={df.columns[0]: 'frame', df.columns[1]: 'player_id', df.columns[2]: 'in_action'})
            
            # Count the number of frames where the sum of in_action is greater than 0
            frame_action_freq = df.groupby('frame')['in_action'].sum()
            # Combine the frequencies
            combined_frame_action_freq = combined_frame_action_freq.add(frame_action_freq, fill_value=0)

            # Get all the frame sequences where the in_action is greater than 0
            in_action_frames_list = frame_action_freq[frame_action_freq > 0].index.to_list()
            # Get all the player ids where the in_action is greater than 0
            in_action_ids = [int(id) for id in df[df['in_action'] > 0]['player_id'].unique()]
            # Find groups of ascending sequences from the in_action_frames
            in_action_sequences = []
            for k, g in groupby(enumerate(in_action_frames_list), lambda ix: ix[0] - ix[1]):
                in_action_sequences.append(list(map(itemgetter(1), g)))
            
            # Get the length of the longest sequence
            longest_sequence = max(in_action_sequences, key=len, default=[])
            # Get the length of the longest sequence that contains frame 24, 25, or 26
            longest_sequence_with_24_25_26 = max([sequence for sequence in in_action_sequences if 24 in sequence or 25 in sequence or 26 in sequence], key=len, default=[])
            # Store the length of the longest sequence in new dictionary
            longest_sequence_dict[skeleton_file] = {'in_action_frames': in_action_frames_list, 'in_action_ids': in_action_ids, 'longest_sequence': longest_sequence, 'longest_sequence_with_24_25_26': longest_sequence_with_24_25_26}

            # Filter sequences that contain frames from certain ranges
            filtered_sequences_24_26 = [sequence for sequence in in_action_sequences if all(frame in sequence for frame in range(24, 27))]
            filtered_sequences_22_28 = [sequence for sequence in in_action_sequences if all(frame in sequence for frame in range(22, 29))]
            filtered_sequences_20_30 = [sequence for sequence in in_action_sequences if all(frame in sequence for frame in range(20, 31))]
            filtered_sequences_15_35 = [sequence for sequence in in_action_sequences if all(frame in sequence for frame in range(15, 36))]
            filtered_sequences[skeleton_file] = {'filtered_sequences_24_26': filtered_sequences_24_26, 'filtered_sequences_22_28': filtered_sequences_22_28, 'filtered_sequences_20_30': filtered_sequences_20_30, 'filtered_sequences_15_35': filtered_sequences_15_35}

            # Get all the frames in certain ranges where the in_action is greater than 0
            in_action_frames_24_26 = [frame for frame in in_action_frames_list if 24 <= frame <= 26]
            in_action_frames_22_28 = [frame for frame in in_action_frames_list if 22 <= frame <= 28]
            in_action_frames_20_30 = [frame for frame in in_action_frames_list if 20 <= frame <= 30]
            in_action_frames_15_35 = [frame for frame in in_action_frames_list if 15 <= frame <= 35]
            missing_frames_24_26 = [frame for frame in range(24, 27) if frame not in in_action_frames_24_26]
            missing_frames_22_28 = [frame for frame in range(22, 29) if frame not in in_action_frames_22_28]
            missing_frames_20_30 = [frame for frame in range(20, 31) if frame not in in_action_frames_20_30]
            missing_frames_15_35 = [frame for frame in range(15, 36) if frame not in in_action_frames_15_35]
            one_skip, two_skip, more_than_two_skip = count_skips(in_action_frames_24_26)
            skip_counts_24_26 = {'one_skip': one_skip, 'two_skip': two_skip, 'more_than_two_skip': more_than_two_skip}
            one_skip, two_skip, more_than_two_skip = count_skips(in_action_frames_22_28)
            skip_counts_22_28 = {'one_skip': one_skip, 'two_skip': two_skip, 'more_than_two_skip': more_than_two_skip}
            one_skip, two_skip, more_than_two_skip = count_skips(in_action_frames_20_30)
            skip_counts_20_30 = {'one_skip': one_skip, 'two_skip': two_skip, 'more_than_two_skip': more_than_two_skip}
            one_skip, two_skip, more_than_two_skip = count_skips(in_action_frames_15_35)
            skip_counts_15_35 = {'one_skip': one_skip, 'two_skip': two_skip, 'more_than_two_skip': more_than_two_skip}
            in_action_frames[skeleton_file] = {'in_action_frames_24_26': in_action_frames_24_26, 'in_action_frames_22_28': in_action_frames_22_28, 'in_action_frames_20_30': in_action_frames_20_30, 'in_action_frames_15_35': in_action_frames_15_35, 'missing_frames_24_26': missing_frames_24_26, 'missing_frames_22_28': missing_frames_22_28, 'missing_frames_20_30': missing_frames_20_30, 'missing_frames_15_35': missing_frames_15_35, 'skip_counts_24_26': skip_counts_24_26, 'skip_counts_22_28': skip_counts_22_28, 'skip_counts_20_30': skip_counts_20_30, 'skip_counts_15_35': skip_counts_15_35, 'no_action_ids_24_26': [], 'no_action_ids_22_28': [], 'no_action_ids_20_30': [], 'no_action_ids_15_35': []}

            # Get all player ids unique regardless of in_action
            player_ids = [int(id) for id in df['player_id'].unique()]
            no_action_ids = [id for id in player_ids if id not in in_action_ids]
            # For each of the no_action_ids, get the frames where the player is detected
            for id in no_action_ids:
                no_action_frames_24_26 = df[(df['player_id'] == id) & (24 <= df['frame']) & (df['frame'] <= 26)]['frame'].to_list()
                no_action_frames_22_28 = df[(df['player_id'] == id) & (22 <= df['frame']) & (df['frame'] <= 28)]['frame'].to_list()
                no_action_frames_20_30 = df[(df['player_id'] == id) & (20 <= df['frame']) & (df['frame'] <= 30)]['frame'].to_list()
                no_action_frames_15_35 = df[(df['player_id'] == id) & (15 <= df['frame']) & (df['frame'] <= 35)]['frame'].to_list()
                # Check for each range if the player is detected in all frames
                if len(no_action_frames_24_26) == 3: in_action_frames[skeleton_file]['no_action_ids_24_26'].append(id)
                if len(no_action_frames_22_28) == 7: in_action_frames[skeleton_file]['no_action_ids_22_28'].append(id)
                if len(no_action_frames_20_30) == 11: in_action_frames[skeleton_file]['no_action_ids_20_30'].append(id)
                if len(no_action_frames_15_35) == 21: in_action_frames[skeleton_file]['no_action_ids_15_35'].append(id)

        # Dump the dictionary to a JSON file
        combined_freq_dict = combined_frame_action_freq.to_dict()
        with open('{}_action_frequencies.json'.format(action), 'w') as json_file:
            json.dump(combined_freq_dict, json_file)
        with open('{}_action_sequences.json'.format(action), 'w') as json_file:
            json.dump(longest_sequence_dict, json_file)
        combined_filtered_sequences = {}
        combined_filtered_sequences['files'] = filtered_sequences
        combined_filtered_sequences['24_26_total'] = sum(1 for data in filtered_sequences.values() if 'filtered_sequences_24_26' in data and data['filtered_sequences_24_26'] != [])
        combined_filtered_sequences['22_28_total'] = sum(1 for data in filtered_sequences.values() if 'filtered_sequences_22_28' in data and data['filtered_sequences_22_28'] != [])
        combined_filtered_sequences['20_30_total'] = sum(1 for data in filtered_sequences.values() if 'filtered_sequences_20_30' in data and data['filtered_sequences_20_30'] != [])
        combined_filtered_sequences['15_35_total'] = sum(1 for data in filtered_sequences.values() if 'filtered_sequences_15_35' in data and data['filtered_sequences_15_35'] != [])
        with open('{}_sequences_in_range.json'.format(action), 'w') as json_file:
            json.dump(combined_filtered_sequences, json_file)
        combined_in_action_frames = {}
        combined_in_action_frames['files'] = in_action_frames
        skip_total_24_26 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
        skip_total_22_28 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
        skip_total_20_30 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
        skip_total_15_35 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
        for data in in_action_frames.values():
            if 'skip_counts_24_26' in data:
                skip_total_24_26 = {key: skip_total_24_26[key] + data['skip_counts_24_26'][key] for key in skip_total_24_26}
            if 'skip_counts_22_28' in data:
                skip_total_22_28 = {key: skip_total_22_28[key] + data['skip_counts_22_28'][key] for key in skip_total_22_28}
            if 'skip_counts_20_30' in data:
                skip_total_20_30 = {key: skip_total_20_30[key] + data['skip_counts_20_30'][key] for key in skip_total_20_30}
            if 'skip_counts_15_35' in data:
                skip_total_15_35 = {key: skip_total_15_35[key] + data['skip_counts_15_35'][key] for key in skip_total_15_35}
        combined_in_action_frames['skip_total_24_26'] = skip_total_24_26
        combined_in_action_frames['skip_total_22_28'] = skip_total_22_28
        combined_in_action_frames['skip_total_20_30'] = skip_total_20_30
        combined_in_action_frames['skip_total_15_35'] = skip_total_15_35
        combined_in_action_frames['no_action_total_24_26'] = sum(len(data['no_action_ids_24_26']) for data in in_action_frames.values())
        combined_in_action_frames['no_action_total_22_28'] = sum(len(data['no_action_ids_22_28']) for data in in_action_frames.values())
        combined_in_action_frames['no_action_total_20_30'] = sum(len(data['no_action_ids_20_30']) for data in in_action_frames.values())
        combined_in_action_frames['no_action_total_15_35'] = sum(len(data['no_action_ids_15_35']) for data in in_action_frames.values())
        with open('{}_action_in_range.json'.format(action), 'w') as json_file:
            json.dump(combined_in_action_frames, json_file)
else:
    ids_length = []
    filename_by_id_counts = {i: [] for i in range(0, 8)} # store the filename of the videos with the same id counts (for presentation purposes)
    all_filtered_sequences = {key: 0 for key in ['24_26_total', '22_28_total', '20_30_total', '15_35_total']}
    skip_total_24_26 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
    skip_total_22_28 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
    skip_total_20_30 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
    skip_total_15_35 = {key: 0 for key in ['one_skip', 'two_skip', 'more_than_two_skip']}
    all_in_action_frames = {key: 0 for key in ['no_action_total_24_26', 'no_action_total_22_28', 'no_action_total_20_30', 'no_action_total_15_35']}
    all_in_action_frames.update({'skip_total_24_26': skip_total_24_26, 'skip_total_22_28': skip_total_22_28, 'skip_total_20_30': skip_total_20_30, 'skip_total_15_35': skip_total_15_35})

    for action_iter in ['Dribble', 'Shoot', 'Pass']:
        with open('{}_action_frequencies.json'.format(action_iter), 'r') as json_file:
            action_frequencies = pd.Series(json.load(json_file))
        with open('{}_action_sequences.json'.format(action_iter), 'r') as json_file:
            longest_sequence_dict = json.load(json_file)
        with open('{}_sequences_in_range.json'.format(action_iter), 'r') as json_file:
            filtered_sequences = json.load(json_file)
        with open('{}_action_in_range.json'.format(action_iter), 'r') as json_file:
            in_action_frames = json.load(json_file)

        # Combine the frequencies
        combined_frame_action_freq = combined_frame_action_freq.add(action_frequencies, fill_value=0)
        # Count the length of in_action_ids and store the length in new pandas series
        for file, data in longest_sequence_dict.items():
            # print(f'{action_iter} - {file}: {len(data["in_action_ids"])}')
            ids_length.append(len(data['in_action_ids']))
            if len(filename_by_id_counts[len(data['in_action_ids'])]) < 7:
                filename_by_id_counts[len(data['in_action_ids'])].append(f'{action_iter} - {file}')

        all_filtered_sequences['24_26_total'] += filtered_sequences['24_26_total']
        all_filtered_sequences['22_28_total'] += filtered_sequences['22_28_total']
        all_filtered_sequences['20_30_total'] += filtered_sequences['20_30_total']
        all_filtered_sequences['15_35_total'] += filtered_sequences['15_35_total']
        print(f'{action_iter} - Total of 24_26 sequences: {filtered_sequences["24_26_total"]}')
        print(f'{action_iter} - Total of 22_28 sequences: {filtered_sequences["22_28_total"]}')
        print(f'{action_iter} - Total of 20_30 sequences: {filtered_sequences["20_30_total"]}')
        print(f'{action_iter} - Total of 15_35 sequences: {filtered_sequences["15_35_total"]}')

        all_in_action_frames['no_action_total_24_26'] += in_action_frames['no_action_total_24_26']
        all_in_action_frames['no_action_total_22_28'] += in_action_frames['no_action_total_22_28']
        all_in_action_frames['no_action_total_20_30'] += in_action_frames['no_action_total_20_30']
        all_in_action_frames['no_action_total_15_35'] += in_action_frames['no_action_total_15_35']
        print(f'{action_iter} - no_action players 24_26: {in_action_frames["no_action_total_24_26"]}')
        print(f'{action_iter} - no_action players 22_28: {in_action_frames["no_action_total_22_28"]}')
        print(f'{action_iter} - no_action players 20_30: {in_action_frames["no_action_total_20_30"]}')
        print(f'{action_iter} - no_action players 15_35: {in_action_frames["no_action_total_15_35"]}')
        all_in_action_frames['skip_total_24_26'] = {key: all_in_action_frames['skip_total_24_26'][key] + in_action_frames['skip_total_24_26'][key] for key in all_in_action_frames['skip_total_24_26']}
        all_in_action_frames['skip_total_22_28'] = {key: all_in_action_frames['skip_total_22_28'][key] + in_action_frames['skip_total_22_28'][key] for key in all_in_action_frames['skip_total_22_28']}
        all_in_action_frames['skip_total_20_30'] = {key: all_in_action_frames['skip_total_20_30'][key] + in_action_frames['skip_total_20_30'][key] for key in all_in_action_frames['skip_total_20_30']}
        all_in_action_frames['skip_total_15_35'] = {key: all_in_action_frames['skip_total_15_35'][key] + in_action_frames['skip_total_15_35'][key] for key in all_in_action_frames['skip_total_15_35']}
        print(f'{action_iter} - skip counts 24_26: {in_action_frames["skip_total_24_26"]}')
        print(f'{action_iter} - skip counts 22_28: {in_action_frames["skip_total_22_28"]}')
        print(f'{action_iter} - skip counts 20_30: {in_action_frames["skip_total_20_30"]}')
        print(f'{action_iter} - skip counts 15_35: {in_action_frames["skip_total_15_35"]}')

    # Print the filenames of the videos with the same id counts
    for id_count in filename_by_id_counts:
        print(f'ID Counts: {id_count} >>')
        for filename in filename_by_id_counts[id_count]:
            print(filename)

    print(f'Total of 24_26 sequences: {all_filtered_sequences["24_26_total"]}')
    print(f'Total of 22_28 sequences: {all_filtered_sequences["22_28_total"]}')
    print(f'Total of 20_30 sequences: {all_filtered_sequences["20_30_total"]}')
    print(f'Total of 15_35 sequences: {all_filtered_sequences["15_35_total"]}')
    print(f'Total no_action players 24_26: {all_in_action_frames["no_action_total_24_26"]}')
    print(f'Total no_action players 22_28: {all_in_action_frames["no_action_total_22_28"]}')
    print(f'Total no_action players 20_30: {all_in_action_frames["no_action_total_20_30"]}')
    print(f'Total no_action players 15_35: {all_in_action_frames["no_action_total_15_35"]}')
    print(f'Total skip counts 24_26: {all_in_action_frames["skip_total_24_26"]}')
    print(f'Total skip counts 22_28: {all_in_action_frames["skip_total_22_28"]}')
    print(f'Total skip counts 20_30: {all_in_action_frames["skip_total_20_30"]}')
    print(f'Total skip counts 15_35: {all_in_action_frames["skip_total_15_35"]}')

    with open('All_action_frequencies.json', 'w') as json_file:
        json.dump(combined_frame_action_freq.to_dict(), json_file)

if not skip_plottings:
    # Define colormap
    blues_cmap = plt.get_cmap('Blues')
    reds_cmap = plt.get_cmap('Reds')

    if action == 'All':
        ### ON-THE-BALL PLAYER PER FRAME ###
        # Generate colors based on the specified ranges
        colors = []
        for i in range(len(combined_frame_action_freq)):
            if 14 <= i <= 18:
                if i == 17: colors.append('blue')
                colors.append(reds_cmap(0.4))
            elif 19 <= i <= 29:
                colors.append(reds_cmap(0.8))
            elif 30 <= i <= 34:
                colors.append(reds_cmap(0.4))
            else:
                colors.append(blues_cmap(0.6))  # Default to the start of the colormap for out-of-range indices
        # Plot the frequency
        plt.figure(figsize=(7, 4))
        plt.bar(combined_frame_action_freq.index, combined_frame_action_freq.values, color=colors)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('On-the-ball Player', fontsize=12)
        # plt.title(f'Detected In-Action Player per Frame for {action} Videos')
        plt.xticks(ticks=range(4, len(combined_frame_action_freq), 5), fontsize=12) # print x-axis label every 5th interval starting from 0
        plt.yticks(ticks=range(250, 3100, 250), fontsize=12, rotation=45) # print y-axis label every 250 units starting from 0
        # plt.yticks(ticks=range(700, 1010, 25)) # print y-axis label every 25 units starting from 700
        plt.grid(axis='y')
        threshold_value = 3018 # 3018 for all, 1006 for else
        # max_value = max(combined_frame_action_freq.values)
        frame_num, max_value = combined_frame_action_freq.idxmax(), combined_frame_action_freq.max()
        success_rate = round(sum(combined_frame_action_freq.values) / (threshold_value * len(combined_frame_action_freq)) * 100, 2)
        line1 = plt.axhline(y=threshold_value, color='r', linestyle='--', label=f'Total Video Clips: {threshold_value}')
        line2 = plt.axhline(y=max_value, color='b', linestyle='--', label=f'Highest Detection: {int(max_value)} (Frame {frame_num})')
        rate_legend = Line2D([0], [0], color='none', linestyle='None', label=f'Success Rate: {success_rate}%')
        plt.legend(handles=[line1, line2, rate_legend], loc='lower right', fontsize=12)
        # plt.ylim(700, 1010)
        plt.ylim(1000, 3100)
        plt.tight_layout()
        plt.show()

        ### ID COUNT PER VIDEO ###
        length_counts = Counter(ids_length)
        lengths = list(length_counts.keys())
        frequencies = list(length_counts.values())
        plt.figure(figsize=(7, 4))
        plt.bar(lengths, frequencies, color=blues_cmap(0.6))
        plt.xlabel('ID Counts', fontsize=12)
        plt.ylabel('Num of Video', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=45)
        # Add text annotations on top of each bar
        for length, frequency in zip(lengths, frequencies):
            plt.text(length, frequency, str(frequency), ha='center', va='bottom', fontsize=10)
        # switch_rate = round(sum(combined_frame_action_freq.values) / (threshold_value * len(combined_frame_action_freq)) * 100, 2)
        # rate_legend = Line2D([0], [0], color='g', linestyle='-', label=f'ID Switch Rate: {switch_rate}%')
        # plt.legend(handles=[rate_legend], loc='upper right')
        plt.tight_layout()
        plt.show()

    else:
        ### IN-ACTIONS FRAME PER VIDEO ###
        # Count the number of in-action frames per video
        in_action_frames_per_video = {i: 0 for i in range(1, 51)}
        longest_sequence_per_video = {i: 0 for i in range(1, 51)}
        longest_sequence_with_24_25_26_per_video = {i: 0 for i in range(1, 51)}
        for file, data in longest_sequence_dict.items():
            length = len(data['in_action_frames'])
            if 1 <= length <= 50:
                in_action_frames_per_video[length] += 1
            length = len(data['longest_sequence'])
            if 1 <= length <= 50:
                longest_sequence_per_video[length] += 1
            length = len(data['longest_sequence_with_24_25_26'])
            if 1 <= length <= 50:
                longest_sequence_with_24_25_26_per_video[length] += 1

        # save the data to a pandas xls file
        in_action_frames_per_video_series = pd.Series(in_action_frames_per_video)
        longest_sequence_per_video_series = pd.Series(longest_sequence_per_video)
        longest_sequence_with_24_25_26_per_video_series = pd.Series(longest_sequence_with_24_25_26_per_video)
        # in_action_frames_per_video_series.to_excel('{}_in_action_frames_per_video.csv'.format(action))
        # longest_sequence_per_video_series.to_excel('{}_longest_sequence_per_video.csv'.format(action))
        # longest_sequence_with_24_25_26_per_video_series.to_excel('{}_longest_sequence_with_24_25_26_per_video.csv'.format(action))

        # Plot the frequency
        plt.bar(in_action_frames_per_video_series.index, in_action_frames_per_video_series.values, color=blues_cmap(0.6))
        plt.xlabel('In-Action Frame Count')
        plt.ylabel('Num of Video')
        plt.title(f'Number of In-Action Frames per {action} Video')
        plt.xticks(ticks=range(5, 51, 5))
        plt.grid(axis='y')
        plt.show()
        # Plot the pie chart for in_action_frames_per_video_series
        aggregated_data = {'50 frames': 0,'>=40 frames': 0,'>=30 frames': 0,'>=20 frames': 0,'>=10 frames': 0,'<10 frames': 0}
        for key, value in in_action_frames_per_video_series.items():
            if key < 10: aggregated_data['<10 frames'] += value
            elif key < 20: aggregated_data['>=10 frames'] += value
            elif key < 30: aggregated_data['>=20 frames'] += value
            elif key < 40: aggregated_data['>=30 frames'] += value
            elif key < 50: aggregated_data['>=40 frames'] += value
            elif key == 50: aggregated_data['50 frames'] += value
        labels = [f'{key}: {value}' for key, value in aggregated_data.items()]
        plt.figure(figsize=(10, 7))
        plt.pie(aggregated_data.values(), labels=labels, startangle=140, autopct='%1.1f%%')
        plt.title(f'Number of In-Action Frames per {action} Video')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

        # Plot the frequency
        plt.bar(longest_sequence_per_video_series.index, longest_sequence_per_video_series.values, color=blues_cmap(0.6))
        plt.xlabel('Sequence Length')
        plt.ylabel('Num of Video')
        plt.title(f'Longest Sequence per {action} Video')
        plt.xticks(ticks=range(5, 51, 5))
        plt.grid(axis='y')
        plt.show()
        # Plot the pie chart for in_action_frames_per_video_series
        aggregated_data = {'50 frames': 0,'>=40 frames': 0,'>=30 frames': 0,'>=20 frames': 0,'>=10 frames': 0,'<10 frames': 0}
        for key, value in longest_sequence_per_video_series.items():
            if key < 10: aggregated_data['<10 frames'] += value
            elif key < 20: aggregated_data['>=10 frames'] += value
            elif key < 30: aggregated_data['>=20 frames'] += value
            elif key < 40: aggregated_data['>=30 frames'] += value
            elif key < 50: aggregated_data['>=40 frames'] += value
            elif key == 50: aggregated_data['50 frames'] += value
        labels = [f'{key}: {value}' for key, value in aggregated_data.items()]
        plt.figure(figsize=(10, 7))
        plt.pie(aggregated_data.values(), labels=labels, startangle=140, autopct='%1.1f%%')
        plt.title(f'Longest Sequence per {action} Video')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

        # Plot the frequency
        plt.bar(longest_sequence_with_24_25_26_per_video_series.index, longest_sequence_with_24_25_26_per_video_series.values, color=blues_cmap(0.6))
        plt.xlabel('Sequence Length')
        plt.ylabel('Num of Video')
        plt.title(f'Longest Sequence with Frame 24, 25, or 26 per {action} Video')
        plt.xticks(ticks=range(5, 51, 5))
        plt.grid(axis='y')
        plt.show()
        # Plot the pie chart for in_action_frames_per_video_series
        aggregated_data = {'50 frames': 0,'>=40 frames': 0,'>=30 frames': 0,'>=20 frames': 0,'>=10 frames': 0,'<10 frames': 0}
        for key, value in longest_sequence_with_24_25_26_per_video_series.items():
            if key < 10: aggregated_data['<10 frames'] += value
            elif key < 20: aggregated_data['>=10 frames'] += value
            elif key < 30: aggregated_data['>=20 frames'] += value
            elif key < 40: aggregated_data['>=30 frames'] += value
            elif key < 50: aggregated_data['>=40 frames'] += value
            elif key == 50: aggregated_data['50 frames'] += value
        labels = [f'{key}: {value}' for key, value in aggregated_data.items()]
        plt.figure(figsize=(10, 7))
        plt.pie(aggregated_data.values(), labels=labels, startangle=140, autopct='%1.1f%%')
        plt.title(f'Longest Sequence with Frame 24, 25, or 26 per {action} Video')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()