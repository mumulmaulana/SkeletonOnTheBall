import argparse
import cv2
import os
import numpy as np
import pandas as pd
import pickle
import sys
from tqdm import tqdm
from module.simpleHRNet.main import SimpleHRNet
from module.simpleHRNet.misc.visualization import get_points_and_skeleton_color, draw_id, draw_points_and_skeleton_v2, joints_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from customutils.utils_skeleton_plot import normalize_skeletons

def draw_image_and_key_lists(input_img, joints, frame_pos=-1, boxes=None):
    """
    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        frame_pos: frame position in the video. If not a video, set to -1.

    Returns:
        Image with skeleton and keypoint lists drawn.
    """
    person_ids = boxes[:, 4] if boxes is not None else np.ones(len(joints), dtype=np.int32)
    key_lists = []
    image = input_img.copy() # in case joints is empty, copy the original image
    for i, (pt, pid) in enumerate(zip(joints, person_ids)):
        skeleton_color, point_color = get_points_and_skeleton_color(skeleton_color_palette='jet', person_index=pid)
        image = draw_points_and_skeleton_v2(input_img, pt, joints_dict()['coco']['skeleton'], skeleton_color, point_color, confidence_threshold=0.3)
        image = draw_id(image, pt[0], skeleton_color, person_index=pid)
        key_lists.append([frame_pos] + [pid] + [0] + pt.flatten().tolist())
    return image, key_lists

def export_keypoints(args, model, action):
    all_files = os.listdir(args.folder_path)
    if args.from_video:
        files = [file for file in all_files if file.lower().endswith(('.mp4', '.avi', '.mkv'))]
    else:
        files = all_files
    annotations = []

    total = len(files)
    with tqdm(total=len(files)) as pbar:
        for iter, file in enumerate(files):
            filepath = os.path.join(args.folder_path, file)
            pbar.update(1)
            pbar.set_description(f"({iter+1}/{total}) Processing {file}")
            if args.from_video:
                filename = os.path.splitext(file)[0]
                if args.save_path is not None:
                    # print('Creating %s ...' % os.path.join(args.save_path, filename+'_HRNet.mp4'))
                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # out = cv2.VideoWriter(os.path.join(args.save_path, filename+'_HRNet.mp4'), fourcc, 25.0, (1280,720))
                    df_keylists = []
                    df_bboxes = []
                cap = cv2.VideoCapture(filepath)
                while True:
                    ret, image = cap.read()
                    if not ret: # If the frame was not read (end of video), then
                        break

                    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    # print('\nProcessing frame %d' % frame_pos)
                    boxes, joints = model.predict(image)
                    result_image, key_lists = draw_image_and_key_lists(image, joints, frame_pos, boxes)

                    if args.save_path is not None:
                        # out.write(result_image)
                        df_keylists.extend(key_lists)
                        df_bboxes.extend(boxes)

                    # Display the annotated frame
                    if args.show_display:
                        cv2.putText(result_image, 'Frame: {}'.format(frame_pos), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.imshow('Annotated Frame', result_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                if args.save_path is not None:
                    if not os.path.exists(args.save_path):
                        os.makedirs(args.save_path)
                    # print('%s succesfully created' % os.path.join(args.save_path, filename+'_HRNet.mp4'))
                    # out.release()
                    df = pd.DataFrame(df_keylists)
                    df.to_csv(os.path.join(args.save_path, filename+'.csv'), index=False, header=False)
                    df = pd.DataFrame(df_bboxes, columns=['x1', 'y1', 'x2', 'y2', 'player_id'])
                    df.to_csv(os.path.join(args.save_path, filename+'_bboxes.csv'), index=False)
                model.reset_tracker()
                cap.release()
                if args.show_display: cv2.destroyAllWindows()
            else:
                jpgpath = filepath
                jpgs = [jpg for jpg in os.listdir(jpgpath) if jpg.lower().endswith(('.jpg', '.jpeg', '.png'))]
                df_keylists = []
                df_keylists_with_mirror = []
                for j, jpg in enumerate(jpgs):
                    jpg_path = os.path.join(jpgpath, jpg)
                    image = cv2.imread(jpg_path)  # B,G,R order
                    mirrored_image = cv2.flip(image, 1)

                    joints = model.predict(image)
                    result_image, key_lists = draw_image_and_key_lists(image, joints, j+1)

                    # generate for mirrored frame as well (comment if not needed)
                    mirrored_joints = model.predict(mirrored_image)
                    mirrored_result_image, mirrored_key_lists = draw_image_and_key_lists(mirrored_image, mirrored_joints, j+1)

                    if args.save_path is not None:
                        df_keylists.extend(key_lists)
                        df_keylists_with_mirror.extend(mirrored_key_lists)
                        imgsavepath = os.path.join(jpgpath, 'HRNet')
                        if not os.path.exists(imgsavepath):
                            os.makedirs(imgsavepath)
                        cv2.imwrite(os.path.join(imgsavepath, jpg.replace('.jpg', '_HRNet.jpg')), result_image)
                        # comment below if mirrored images are not needed
                        imgsavepath = os.path.join(jpgpath, 'HRNet_mirror')
                        if not os.path.exists(imgsavepath):
                            os.makedirs(imgsavepath)
                        cv2.imwrite(os.path.join(imgsavepath, jpg.replace('.jpg', '_HRNet_mirror.jpg')), mirrored_result_image)

                    # Display the annotated frame
                    if args.show_display:
                        cv2.imshow('Annotated Frame', result_image)
                        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                        cv2.destroyAllWindows()
                
                anno = dict()
                anno['img_shape'] = (720, 1280)
                anno['original_shape'] = (720, 1280)
                anno['total_frames'] = len(jpgs)
                anno['frame_dir'] = file+'_'+action
                anno['label'] = 1
                df_keylists = np.array(df_keylists)
                keylists_np = df_keylists[:, 3:].reshape(-1, 17, 3)
                keypoint_np = keylists_np[..., :-1]
                keypoint_np = keypoint_np[..., ::-1]  # Convert to (x, y) format
                keypoint_score_np = keylists_np[..., -1]
                keypoint_np = np.expand_dims(keypoint_np, axis=0)
                keypoint_score_np = np.expand_dims(keypoint_score_np, axis=0)
                anno['keypoint'] = keypoint_np
                anno['keypoint_score'] = keypoint_score_np
                normalized_anno = normalize_skeletons([anno], (1280, 720))
                annotations.append(normalized_anno[0])

                # add mirrored keypoints to the annotations (comment if not needed)
                # mirror = dict()
                # mirror['img_shape'] = (720, 1280)
                # mirror['original_shape'] = (720, 1280)
                # mirror['total_frames'] = len(jpgs)
                # mirror['frame_dir'] = file+'_'+action+'_mirror'
                # mirror['label'] = 1
                # df_keylists_with_mirror = np.array(df_keylists_with_mirror)
                # keylists_np = df_keylists_with_mirror[:, 3:].reshape(-1, 17, 3)
                # keypoint_np = keylists_np[..., :-1]
                # keypoint_np = keypoint_np[..., ::-1]  # Convert to (x, y) format
                # keypoint_score_np = keylists_np[..., -1]
                # keypoint_np = np.expand_dims(keypoint_np, axis=0)
                # keypoint_score_np = np.expand_dims(keypoint_score_np, axis=0)
                # mirror['keypoint'] = keypoint_np
                # mirror['keypoint_score'] = keypoint_score_np
                # normalized_mirror = normalize_skeletons([mirror], (1280, 720))
                # annotations.append(normalized_mirror[0])

                if args.save_path is not None:
                    df = pd.DataFrame(df_keylists)
                    df.to_csv(os.path.join(jpgpath, file+'.csv'), index=False, header=False)
                    # comment below if mirrored keypoints are not needed
                    # df = pd.DataFrame(df_keylists_with_mirror)
                    # df.to_csv(os.path.join(jpgpath, file+'_mirror.csv'), index=False, header=False)
    
    return annotations # will be empty if from_video is True

# Call the function with the desired folder path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images.', default='./test_videos')
    parser.add_argument('--weights', type=str, help='Path to the HRNet weights.', default='module/simpleHRNet/weights/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--from_video', action='store_true', help='Set this flag if the files are from a video.')
    parser.add_argument('--show_display', action='store_true', help='Set this flag if you want to display the annotated images.')
    parser.add_argument('--save_path', type=str, help='Path to the folder where the annotated images will be saved.', default=None)
    args = parser.parse_args()

    if not args.from_video:
        base_folder = args.folder_path
        args.save_path = args.folder_path
        model = SimpleHRNet(48, 17, args.weights, yolo_version='v5', yolo_model_def='yolov5l', device='cuda', multiperson=False)

        pkl = dict()
        pkl['annotations'] = []
        for action in ['dribble', 'shoot', 'pass']:
            args.folder_path = os.path.join(base_folder, action)
            # export_keypoints(args, model)
            pkl['annotations'].extend(export_keypoints(args, model, action))

        if args.save_path is not None:
            with open(os.path.join(args.save_path, 'annotations.pkl'), 'wb') as f:
                pickle.dump(pkl, f)
    else:
        model = SimpleHRNet(48, 17, args.weights, yolo_version='v5', yolo_model_def='yolov5l', device='cuda', return_bounding_boxes=True, tracking=True)
        if args.save_path == None and not args.show_display:
            args.save_path = args.folder_path
        export_keypoints(args, model, None)