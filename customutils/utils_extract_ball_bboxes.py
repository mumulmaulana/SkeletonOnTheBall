import argparse
import os
import cv2
import sys
import numpy as np
import supervision as sv
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, '/home/ogatalab/Documents/SoccerProjects/sports')
from sports.common.ball import BallTracker, BallAnnotator

def export_ball_bboxes(folder_path, output_dir=None, save_video=False):
    ball_detection_model = YOLO("/home/ogatalab/Documents/SoccerProjects/sports/examples/soccer/data/football-ball-detection.pt").to(device="cuda")
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    files = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith('.mp4')]
    for file in files:
        print(f'Processing {file}.mp4 ...')
        video_path = os.path.join(folder_path, file + '.mp4')
        if save_video:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out = cv2.VideoWriter(os.path.join(output_dir, file + '_ball.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (1280, 720))

        ball_detections = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret: # If the frame was not read (end of video), then
                break

            detections = slicer(frame).with_nms(threshold=0.1)
            detections = ball_tracker.update(detections)

            # store frame number and ball detections
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            for bbox in detections.xyxy:
                bbox = list(map(int, bbox))
                ball_detections.append([frame_number] + bbox)

            # Annotate the frame with bounding boxes
            annotated_frame = frame.copy()
            annotated_frame = ball_annotator.annotate_bbox(frame, detections)
            cv2.imshow(f'{file}_ball.mp4', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # write the annotated frame to a video file
            if save_video:
                out.write(annotated_frame)
        
        # release the video capture and writer objects
        if save_video:
            out.release()
        cap.release()
        cv2.destroyAllWindows()

        # save the ball bounding boxes
        df = pd.DataFrame(ball_detections, columns=['frame', 'x1', 'y1', 'x2', 'y2'])
        df.to_csv(os.path.join(output_dir, file + '_ballbboxes.csv'), index=False)

    print('Done!')

def exec_handler(folder_path, output_dir, save_anno_videos):
    if folder_path == 'all':
        folder_path = '/media/ogatalab/OgataLab8TB/captured_frame'
        actions = ['Shoot', 'Pass', 'Dribble']
        for action in actions:
            export_ball_bboxes(os.path.join(folder_path, action), os.path.join(folder_path, action + '_HRNet'), save_anno_videos)
    else:
        export_ball_bboxes(folder_path, output_dir, save_anno_videos)

# Call the function with the desired folder path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract ball bounding box from videos.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing videos.', default='./test_videos')
    parser.add_argument('--output_dir', type=str, help='Path to the folder where the annotated videos will be saved.', default='./test_output')
    parser.add_argument('--save_anno_videos', action='store_true', help='Set this flag if you want to save the annotated videos.')
    args = parser.parse_args()

    exec_handler(args.folder_path, args.output_dir, args.save_anno_videos)