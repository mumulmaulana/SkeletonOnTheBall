from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer
import sys
import cv2
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# local modules
from module.simpleHRNet.misc.visualization import joints_dict, get_points_and_skeleton_color, draw_id, draw_bbox, draw_points_and_skeleton_v2

# get current execution path
current_path = os.path.dirname(os.path.realpath(__file__))
interface = uic.loadUiType(os.path.join(current_path, "annotation.ui"))[0]
current_frame_pos = 0 # global variable for video slider, for easier access between MyWindow and VideoThread

# init skeleton joints dictionary
keypoints_dict = joints_dict()['coco']['keypoints']

class VideoThread(QThread):
    send_signal = pyqtSignal(np.ndarray, list)
    receive_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.frame_pos = 0
        self.default_fps = 25

    def load_new_video(self, video_file):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()  # release the old video source
        self.cap = cv2.VideoCapture(video_file)  # load the new video source
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # update the length
        self.frame_pos = 0  # reset the frame position
        self.fps = self.default_fps
        self.filename = video_file

    def play(self):
        ret, cv_img = self.cap.read()
        if ret:
            self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.send_signal.emit(cv_img, [self.filename, self.frame_pos, self.fps, self.default_fps, self.length])
        else:
            self.pause()

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.play)
        self.timer.start(int(1000/self.fps))

    def resume(self):
        global current_frame_pos
        self.frame_pos = current_frame_pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
        if not self.timer.isActive():
            self.timer.start(int(1000/self.fps))

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()

    def stop(self):
        self.timer.stop()
        self.cap.release()

    def skip_forward(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
        self.play() # set the frame to the window

    def skip_backward(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos-2) # because -1 is the same frame
        self.play() # set the frame to the window

    def skip_ten_forward(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos+10)
        self.play() # set the frame to the window

    def skip_ten_backward(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos-12)
        self.play() # set the frame to the window

    def increase_speed(self):
        if self.fps < self.default_fps*8:
            self.timer.stop()
            self.fps *= 2 # double the playback speed
            self.timer.start(int(1000/self.fps))

    def decrease_speed(self):
        if self.fps > self.default_fps:
            self.timer.stop()
            self.fps /= 2 # half the playback speed
            self.timer.start(int(1000/self.fps))


class MyWindow(QMainWindow, interface):
    def __init__(self, args, parent=None):
        super(MyWindow, self).__init__()
        # self.available_cameras = QCameraInfo.availableCameras()  # Getting available cameras

        # setup UI for main window
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        # self.setStyleSheet("background-color: white;")
        # print(QtWidgets.QMainWindow.geometry(self))
        self.resize(1712, 938)
        self.setMaximumSize(1712, 938)
        self.setWindowTitle('Annotation Tool')
        self.initWindow(args.file_path)

########################################################################################################################
#                                                   Windows                                                            #
########################################################################################################################
    def initWindow(self, file_path):
        # Init status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background : lightblue;")  # Setting style sheet to the status bar
        self.setStatusBar(self.status)  # Adding status bar to the main window
        # init video thread
        self.thread = VideoThread()
        # init csv skeleton file
        self.skeleton_file = None

        # use file path from args and collect the filename list
        self.file_path = file_path
        if os.path.isdir(self.file_path):
            # Original video has "HRNet" at the end of the filename. This commented version is for the original video
            # self.filename_list = ['_'.join(os.path.splitext(file)[0].split('_')[:-1]) for file in os.listdir(self.file_path) if file.endswith('.mp4')]
            self.filename_list = [os.path.splitext(file)[0] for file in os.listdir(self.file_path) if file.endswith('.mp4')]
            self.filename_list_len = len(self.filename_list)

            # get task name from the folder name
            self.task = self.file_path.split('/')[-1].split('_')[0]
            self.vidsource_path = self.file_path
            self.video_index = 0

            # init video search section
            self.searchVideo.setText(str(self.video_index+1))
            self.remainingVideos.setText('/{} videos'.format(str(self.filename_list_len)))
            self.searchVideo.returnPressed.connect(self.search_video)
            self.searchVideoButton.clicked.connect(self.search_video)
            self.vidName.returnPressed.connect(self.search_video)

            # init checkbox for skeleton annotation
            self.showSkeletonCheckbox.setChecked(True)
            self.showIDCheckbox.setChecked(True)

            # load the video file
            self.load_video(self.filename_list[self.video_index])
        else:
            self.load_video(self.file_path)
            self.tabWidget_3.currentIndex(0)
            self.tabWidget_3.setTabEnabled(1, False)

########################################################################################################################
#                                                   Buttons                                                            #
########################################################################################################################
    # Activates when Start/Stop video button is clicked to Start (playButton)
    def ClickPlayVideo(self):
        try:
            self.playButton.clicked.disconnect(self.ClickPlayVideo)
            self.playButton.setText('Stop')
            self.thread.send_signal.connect(self.update_image)
            # start the thread
            self.thread.start()
            self.playButton.clicked.connect(self.ClickStopVideo)
            self.pauseButton.clicked.connect(self.ClickPauseVideo)
            # connect the skip increase speed buttons
            self.speedIncr.clicked.connect(self.thread.increase_speed)
            self.speedDecr.clicked.connect(self.thread.decrease_speed)
            # disconnect all frame manipulation buttons
            try: self.nextButton.clicked.disconnect(self.thread.skip_forward)
            except TypeError: pass
            try: self.prevButton.clicked.disconnect(self.thread.skip_backward)
            except TypeError: pass
            try: self.captureButton.clicked.disconnect(self.save_frame)
            except TypeError: pass
            try: self.nextVideoButton.clicked.disconnect(self.nextVideoLambda)
            except TypeError: pass
            try: self.prevVideoButton.clicked.disconnect(self.prevVideoLambda)
            except TypeError: pass
        except Exception as e :
            print(f"Error starting video: {e}")
            self.status.showMessage(f"Error starting video: {e}")

    # Activates when Start/Stop video button is clicked to Stop (playButton)
    def ClickStopVideo(self):
        try:
            self.thread.send_signal.disconnect()
            self.thread.stop()
            self.playButton.clicked.disconnect(self.ClickStopVideo)
            self.playButton.setText('Play')
            try: self.pauseButton.clicked.disconnect(self.ClickPauseVideo)
            except TypeError: pass
            try: self.pauseButton.clicked.disconnect(self.ClickResumeVideo)
            except TypeError: pass
            self.pauseButton.setText('Pause')
            # allow for video navigation
            self.nextVideoLambda = lambda x: self.video_nav(1)
            self.prevVideoLambda = lambda x: self.video_nav(-1)
            self.nextVideoButton.clicked.connect(self.nextVideoLambda)
            self.prevVideoButton.clicked.connect(self.prevVideoLambda)
            # remove all functionalities from the control buttons
            try:
                self.speedIncr.clicked.disconnect(self.thread.increase_speed)
                self.speedDecr.clicked.disconnect(self.thread.decrease_speed)
            except TypeError:
                pass
            try:
                self.speedIncr.clicked.disconnect(self.thread.skip_ten_forward)
                self.speedDecr.clicked.disconnect(self.thread.skip_ten_backward)
            except TypeError:
                pass
            try:
                self.captureSkeletonForCurrentFrame.clicked.disconnect(self.capture_skeleton)
                self.captureSkeletonForAllFrames.clicked.disconnect(self.capture_skeleton)
                self.saveSkeleton.clicked.disconnect(self.save_skeleton_file)
            except TypeError:
                pass
            self.speedIncr.setText('Speed >>')
            self.speedDecr.setText('<< Speed')
            self.init_slider = 0
            # disconnect checkbox listener
            self.showSkeletonCheckbox.stateChanged.disconnect(self.checkbox_state)
            self.showIDCheckbox.stateChanged.disconnect(self.checkbox_state)
            self.maskFrameCheckbox.stateChanged.disconnect(self.checkbox_state)
            self.showBboxCheckbox.stateChanged.disconnect(self.checkbox_state)
            self.showInActionOnlyCheckbox.stateChanged.disconnect(self.checkbox_state)
            self.showSelectedCheckbox.stateChanged.disconnect(self.checkbox_state)
            self.skeletonListWidget.itemClicked.disconnect(self.skeleton_itemClicked)
            self.status.showMessage('Video Stopped. Please load a new video to start again')
        except Exception as e:
            print(f"Error stopping video: {e}")
            self.status.showMessage(f"Error stopping video: {e}")

    # Activates when Pause/Resume video button is clicked to Pause (pauseButton)
    def ClickPauseVideo(self):
        self.pauseButton.clicked.disconnect(self.ClickPauseVideo)
        self.status.showMessage('Video Paused')
        self.pauseButton.setText('Resume')
        self.thread.pause()
        # self.pauseButton.clicked.connect(self.thread.resume)
        self.pauseButton.clicked.connect(self.ClickResumeVideo)
        # connect the skip frame and capture frame buttons
        self.nextButton.clicked.connect(self.thread.skip_forward)
        self.prevButton.clicked.connect(self.thread.skip_backward)
        self.captureButton.clicked.connect(self.save_frame)
        self.captureSkeletonForCurrentFrame.clicked.connect(self.capture_skeleton)
        self.captureSkeletonForAllFrames.clicked.connect(self.capture_skeleton)
        self.saveSkeleton.clicked.connect(self.save_skeleton_file)
        self.speedIncr.clicked.disconnect(self.thread.increase_speed)
        self.speedDecr.clicked.disconnect(self.thread.decrease_speed)
        self.speedIncr.clicked.connect(self.thread.skip_ten_forward)
        self.speedDecr.clicked.connect(self.thread.skip_ten_backward)
        self.speedIncr.setText('Skip >>')
        self.speedDecr.setText('<< Skip')

    # Activates when Pause/Resume video button is clicked to Resume (pauseButton)
    def ClickResumeVideo(self):
        self.pauseButton.clicked.disconnect(self.ClickResumeVideo)
        self.pauseButton.setText('Pause')
        self.thread.resume()
        # self.pauseButton.clicked.disconnect(self.thread.resume)
        self.pauseButton.clicked.connect(self.ClickPauseVideo)
        # disconnect the skip frame buttons
        self.nextButton.clicked.disconnect(self.thread.skip_forward)
        self.prevButton.clicked.disconnect(self.thread.skip_backward)
        self.captureButton.clicked.disconnect(self.save_frame)
        self.captureSkeletonForCurrentFrame.clicked.disconnect(self.capture_skeleton)
        self.captureSkeletonForAllFrames.clicked.disconnect(self.capture_skeleton)
        # self.saveSkeleton.clicked.disconnect(self.save_skeleton_file)
        self.speedIncr.clicked.disconnect(self.thread.skip_ten_forward)
        self.speedDecr.clicked.disconnect(self.thread.skip_ten_backward)
        self.speedIncr.setText('Speed >>')
        self.speedDecr.setText('<< Speed')
        self.speedIncr.clicked.connect(self.thread.increase_speed)
        self.speedDecr.clicked.connect(self.thread.decrease_speed)

########################################################################################################################
#                                                   Actions                                                            #
########################################################################################################################
    def search_video(self):
        try:
            sender = self.sender()
            if sender == self.searchVideoButton or sender == self.searchVideo:
                video_index = int(self.searchVideo.text()) - 1
            elif sender == self.vidName:
                video_index = self.filename_list.index(self.vidName.text().strip())
            if video_index >= 0 and video_index < self.filename_list_len:
                self.video_index = video_index
                self.load_video(self.filename_list[self.video_index])
                self.searchVideo.setText(str(self.video_index+1))
            else:
                self.status.showMessage('Invalid video index. Please input a number between 1 and %d' % self.filename_list_len)
        except Exception as e:
            print(f"Error searching video: {e}")
            self.status.showMessage(f"Error searching video: {e}")
    
    def load_video(self, video_file):
        try:
            self.thread.send_signal.disconnect()
        except Exception:
            pass  # ignore error if signal was not connected
        if len(video_file.split('/')) > 1:
            self.thread.load_new_video(video_file)
        else:
            self.thread.load_new_video(os.path.join(self.vidsource_path, video_file+'.mp4'))

            # load the skeleton and bounding boxes file
            if self.skeleton_file is not None:
                self.skeleton_file = None
                self.bbox_file = None
                # self.ballbbox_file = None
                self.augmented_file = None
            self.skeleton_file = pd.read_csv(os.path.join(self.file_path, video_file+'.csv'), header=None)
            self.skeleton_file = self.skeleton_file.rename(columns={self.skeleton_file.columns[0]: 'frame', self.skeleton_file.columns[1]: 'player_id', self.skeleton_file.columns[2]: 'in_action'})
            self.bbox_file = pd.read_csv(os.path.join(self.file_path, video_file+'_bboxes.csv'))
            self.bbox_file['frame'] = self.skeleton_file['frame'].values
            # self.ballbbox_file = pd.read_csv(os.path.join(self.file_path, video_file+'_ballbboxes.csv'))

            # augmented (uncomment when in needs to display augmented results)
            # self.augmented_file = pd.read_csv(os.path.join(self.file_path, video_file+'_augmented.csv'), header=None)
            # self.augmented_file = self.augmented_file.drop(columns=self.augmented_file.columns[3:7])
            # self.augmented_file.columns = range(self.augmented_file.shape[1]) # Reset the column indices
            # self.augmented_file = self.augmented_file.rename(columns={self.augmented_file.columns[0]: 'frame', self.augmented_file.columns[1]: 'player_id', self.augmented_file.columns[2]: 'in_action'})
            # interleaved_rows = []
            # for frame in range(self.skeleton_file['frame'].min(), self.skeleton_file['frame'].max() + 1):
            #     skeleton_rows = self.skeleton_file[self.skeleton_file['frame'] == frame]
            #     augmented_rows = self.augmented_file[self.augmented_file['frame'] == frame]
            #     interleaved_rows.append(skeleton_rows)
            #     interleaved_rows.append(augmented_rows)
            # self.skeleton_file = pd.concat(interleaved_rows, ignore_index=True)

            self.source_img = None
            self.player_id = None
            is_annotated_row = self.skeleton_file.loc[self.skeleton_file['in_action'] == 1, 'player_id'] # get the player_id in action to see if this video has been annotated
            if not is_annotated_row.empty:
                self.is_saved = True
            else:
                self.is_saved = False

            # update the skeleton action label
            self.skeleton_action_label_update()

            # Listener for the skeleton anno tab
            self.showSkeletonCheckbox.stateChanged.connect(self.checkbox_state)
            self.showIDCheckbox.stateChanged.connect(self.checkbox_state)
            self.maskFrameCheckbox.stateChanged.connect(self.checkbox_state)
            self.showBboxCheckbox.stateChanged.connect(self.checkbox_state)
            self.showInActionOnlyCheckbox.stateChanged.connect(self.checkbox_state)
            self.showSelectedCheckbox.stateChanged.connect(self.checkbox_state)
            self.skeletonListWidget.itemClicked.connect(self.skeleton_itemClicked)

        # init slider for the first time
        self.init_slider = 0

        # connect button to start and pause video
        self.playButton.clicked.connect(self.ClickPlayVideo)

        # update status bar
        self.vidName.setText(video_file)
        self.status.showMessage('%s Loaded. Ready to start' % video_file)

        # Set Frame
        pixmap = QPixmap(self.imageLabel.size())
        pixmap.fill(Qt.black)
        self.imageLabel.setPixmap(pixmap)

    def capture_skeleton(self):
        if self.player_id is not None:
            if self.sender() == self.captureSkeletonForCurrentFrame:
                # reset all in_action to 0 for current frame
                self.skeleton_file.loc[self.skeleton_file['frame'] >= current_frame_pos, 'in_action'] = 0
                # assign the selected player_id 'in_action' to 1 only if the player_id is empty or not the same as the current player_id
                if self.player_id_row.empty or self.player_id != self.player_id_row.iloc[0].astype(int):
                    self.skeleton_file.loc[(self.skeleton_file['frame'] >= current_frame_pos) & (self.skeleton_file['player_id'] == self.player_id), 'in_action'] = 1
                    self.captureSkeletonForCurrentFrame.setText("Remove ID")
                else:
                    self.captureSkeletonForCurrentFrame.setText("Capture ID")
            elif self.sender() == self.captureSkeletonForAllFrames:
                # reset all in_action to 0 for all frames
                self.skeleton_file['in_action'] = 0
                # assign the selected player_id 'in_action' to 1
                self.skeleton_file.loc[self.skeleton_file['player_id'] == self.player_id, 'in_action'] = 1
                if self.player_id_row.empty or self.player_id != self.player_id_row.iloc[0].astype(int):
                    self.skeleton_file.loc[self.skeleton_file['player_id'] == self.player_id, 'in_action'] = 1
                    self.captureSkeletonForAllFrames.setText("Remove ID for All Frames")
                else:
                    self.captureSkeletonForAllFrames.setText("Capture ID for All Frames")
            self.status.showMessage('Skeleton for Player ID %d is captured' % self.player_id)
            self.is_saved = False
            self.skeleton_action_label_update()
        else:
            self.status.showMessage('Please select a player ID first')
    
    def save_skeleton_file(self):
        self.skeletonActionLabel.setText('Saving...')
        if self.skeleton_file is not None:
            self.skeleton_file.to_csv(os.path.join(self.file_path, self.filename_list[self.video_index]+'.csv'), index=False, header=False)
            self.status.showMessage(f'{self.filename_list[self.video_index]}.csv is saved')
            self.is_saved = True
            self.skeleton_action_label_update()
        else:
            self.status.showMessage('No skeleton file to save')

    def skeleton_action_label_update(self):
        if self.skeleton_file is not None:
            if self.skeleton_file['in_action'].sum() > 0:
                if self.is_saved:
                    self.skeletonActionLabel.setStyleSheet("background-color: green;")
                    self.skeletonActionLabel.setText('This video has been annotated')
                    self.showInActionOnlyCheckbox.setChecked(True)
                else:
                    self.skeletonActionLabel.setStyleSheet("background-color: yellow;")
                    self.skeletonActionLabel.setText('Click \'Save Skeleton File\' to save the annotation')
            else:
                self.skeletonActionLabel.setStyleSheet("background-color: red;")
                self.skeletonActionLabel.setText('This video has not been annotated')
        else:
            self.skeletonActionLabel.setStyleSheet("background-color: red;")
            self.skeletonActionLabel.setText('Please load a video first')

    def update_image(self, cv_img, received_signal):
        """Updates the imageLabel with a new opencv image"""
        global current_frame_pos
        self.current_img = cv_img
        self.source_img = self.current_img.copy()
        self.filename_rcv, self.frame_pos_rcv, self.fps_rcv, self.default_fps_rcv, self.length_rcv = received_signal
        current_frame_pos = self.frame_pos_rcv
        # init slider for the first time
        if self.init_slider == 0:
            self.videoSlider.tracking = True
            self.videoSlider.setRange(0, self.length_rcv)
            self.videoSlider.sliderMoved.connect(self.seek_video)
            self.init_slider = 1
        
        if not self.showSelectedCheckbox.isChecked():
            self.player_id = None

        self.draw_frame(self.player_id)
        self.update_control()
        
    def skeleton_itemClicked(self, item):
        self.player_id = int(item.text().split(' ')[2])
        self.jointsConfListWidget.clear()
        if not self.player_id_row.empty and self.player_id == self.player_id_row.iloc[0].astype(int):
            self.captureSkeletonForCurrentFrame.setText("Remove ID")
            self.captureSkeletonForAllFrames.setText("Remove ID for All Frames")
        else:
            self.captureSkeletonForCurrentFrame.setText("Capture ID")
            self.captureSkeletonForAllFrames.setText("Capture ID for All Frames")
        self.draw_frame(self.player_id)

    def checkbox_state(self, state):
        self.draw_frame(self.player_id)

    def get_points_and_skeleton_color(self, skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0):
        try:
            skeleton_colors = np.round(
                np.array(plt.get_cmap(skeleton_color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            skeleton_colors = np.round(
                np.array(plt.get_cmap(skeleton_color_palette)(np.linspace(0, 1, skeleton_palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()

        point_colors = skeleton_colors # use the same color palette as the skeleton

        # get the color for the current person
        return skeleton_colors[person_index % len(skeleton_colors)], point_colors[person_index % len(point_colors)]

    def draw_points_and_skeleton_v2(self, image, points, skeleton, skeleton_colors, point_colors, confidence_threshold=0.5):
        for joint in skeleton:
            pt1, pt2 = points[joint]
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                image = cv2.line(
                    image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                    tuple(skeleton_colors), 2
                )
                image = cv2.circle(image, (int(pt1[1]), int(pt1[0])), 2, tuple(point_colors), -1)
                image = cv2.circle(image, (int(pt2[1]), int(pt2[0])), 2, tuple(point_colors), -1)
        return image
    
    def draw_player_id(self, image, points, skeleton_colors, person_index=0):
        cv2.putText(image, 'ID: %d' % person_index, (int(points[1]), int(points[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(skeleton_colors), 2)
        return image
    
    def draw_player_bbox(self, image, bbox, skeleton_colors):
        x1, y1, x2, y2 = bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), tuple(skeleton_colors), 2)
        return image

    def draw_frame(self, selected_player_id=None):
        if self.source_img is not None:
            self.current_img = self.source_img.copy()
            self.confidenceLabel.setText('Confidence')
            # get necessary skeleton and bbox data
            if self.skeleton_file is not None:
                current_frame_skeletons = self.skeleton_file.loc[self.skeleton_file['frame'] == current_frame_pos].iloc[:, 1:].to_numpy() # get all skeletons for the current frame
                in_action_player_id = self.skeleton_file.loc[(self.skeleton_file['in_action'] == 1) & (self.skeleton_file['frame'] == current_frame_pos), 'player_id'] # get the player_id in action for the current frame
                self.player_id_row = in_action_player_id # update the player_id_row
                player_ids = current_frame_skeletons[:, 0].astype(int) # get all player ids for the current frame
                joints = current_frame_skeletons[:, 2:].reshape(-1, 17, 3) # reshape to 17 keypoints, 3 values (y, x, confidence)

                if selected_player_id is not None:
                    print_skeleton_list = False
                    print_confidence_list = True
                    pid_draw = selected_player_id
                else:
                    if not in_action_player_id.empty:
                        print_skeleton_list = True
                        print_confidence_list = True
                        pid_draw = in_action_player_id.iloc[0].astype(int)
                    else:
                        print_skeleton_list = True
                        print_confidence_list = False
                        pid_draw = None

                # skeleton list
                if print_skeleton_list:
                    self.skeletonListWidget.clear()
                    for i, pid in enumerate(player_ids):
                        action_text = " (in_action)" if not in_action_player_id.empty and pid == in_action_player_id.iloc[0].astype(int) else ""
                        self.skeletonListWidget.addItem(f'Player ID: {pid}{action_text}')
                    items = [self.skeletonListWidget.item(i).text() for i in range(self.skeletonListWidget.count())]
                    if items is not None:
                        sorted_items = sorted(items, key=lambda item: int(item.split(' ')[2]))
                        self.skeletonListWidget.clear()
                        for item in sorted_items:
                            self.skeletonListWidget.addItem(item)

                # confidence list
                self.jointsConfListWidget.clear()
                if print_confidence_list and pid_draw is not None:
                    self.confidenceLabel.setText(f'Confidence (Player ID: {pid_draw}, Frame: {current_frame_pos})')
                    pid_index = np.where(player_ids == pid_draw)[0][0]
                    for i, keypoint in enumerate(joints[pid_index]):
                        keypoint_name = keypoints_dict[i]
                        self.jointsConfListWidget.addItem(f'{keypoint_name}: {keypoint[2]:.2f}')

                # draw the frame
                if self.maskFrameCheckbox.isChecked():
                    self.current_img = np.zeros((self.current_img.shape[0], self.current_img.shape[1], 3), np.uint8)
                for pt, pid in zip(joints, player_ids):
                    if self.showInActionOnlyCheckbox.isChecked() and pid != pid_draw: # skip if showInActionOnlyCheckbox is checked and player_id is not in action
                        continue
                    elif selected_player_id is not None and pid != pid_draw: # skip if selected_player_id is not None and player_id is not the same as selected_player_id
                        continue
                    skeleton_color, point_color = get_points_and_skeleton_color(skeleton_color_palette='jet', person_index=pid)
                    if self.showSkeletonCheckbox.isChecked():
                        self.current_img = draw_points_and_skeleton_v2(self.current_img, pt, joints_dict()['coco']['skeleton'], skeleton_color, point_color, confidence_threshold=0.3)
                    if self.showBboxCheckbox.isChecked():
                        bbox = self.bbox_file.loc[(self.bbox_file['frame'] == current_frame_pos) & (self.bbox_file['player_id'] == pid)].iloc[:, :4].to_numpy()
                        if len(bbox) > 0:
                            self.current_img = draw_bbox(self.current_img, bbox[0], skeleton_color)
                    if self.showIDCheckbox.isChecked():
                        self.current_img = draw_id(self.current_img, pt[0], skeleton_color, person_index=pid)
                # ball_bbox = self.ballbbox_file.loc[self.ballbbox_file['frame'] == current_frame_pos].iloc[:, 1:].to_numpy()
                # if len(ball_bbox) > 0:
                #     self.current_img = draw_bbox(self.current_img, ball_bbox[0], (0, 0, 255))

            qt_img = self.convert_cv_qt(self.current_img, 1280, 720)
            self.imageLabel.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def save_frame(self):
        """Save current frame according to action category"""
        save_filename = '%s_%d.jpg' % (os.path.splitext(self.filename_rcv)[0].split('/')[-1], self.frame_pos_rcv)
        save_folder = os.path.join(args.save_path, self.action.currentText())
        cv2.imwrite(os.path.join(save_folder, save_filename), self.current_img)
        self.status.showMessage('%s is saved on %s folder' % (save_filename, self.action.currentText()))
        self.listWidget.addItem(self.status.currentMessage())
        roi_img = self.convert_cv_qt(self.current_img, 320, 240)
        self.roiImageLabel.setPixmap(roi_img)

    def update_control(self):
        # update slider 
        self.videoSlider.setValue(current_frame_pos)
        # update status message
        status = 'Video Running. Playback Speed %dx' % int(self.fps_rcv/self.default_fps_rcv)
        self.status.showMessage(status)
        self.frameCount.setText('%d/%d' % (current_frame_pos, self.length_rcv))

    def seek_video(self, value):
        global current_frame_pos
        current_frame_pos = value
        self.thread.resume()

    def video_nav(self, incr):
        self.video_index = (self.video_index + incr + self.filename_list_len) % self.filename_list_len
        self.load_video(self.filename_list[self.video_index])
        self.searchVideo.setText(str(self.video_index+1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Annotate video frame.')
    parser.add_argument('file_path', type=str, help='Path to the video file', default='./sample')
    parser.add_argument('--save_path', type=str, help='Path to save the annotated video', default='./captured_frame')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = MyWindow(args)
    win.show()
    sys.exit(app.exec())