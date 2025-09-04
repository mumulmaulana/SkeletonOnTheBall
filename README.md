# Skeleton On-the-ball Annotator

GUI System for annotating On-the-ball players from soccer videos.
Utilizes YOLO for Player Detection, and HRNet for Pose Estimation.
Annotation file is saved in CSV format.

Contains modified code from [Supervision](https://github.com/roboflow/supervision) and [simple-HRNet](https://github.com/stefanopini/simple-HRNet).

<p align="center">
  <img src="demo.gif" width="100%" alt="Project demo GIF">
</p>

---

## Installation

```bash
# clone repo
git clone https://github.com/mumulmaulana/SkeletonOnTheBall.git
cd path/to/this/repo

# install deps
pip install -r requirements.txt
```

## Prerequisites

- Download the required weights for pose estimation model from the [https://github.com/leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
	- The one being used in the demo is [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)
- Put the weights inside `annotate/module/simpleHRNet/weights`

## Usage

Step 1. Put all the desired videos to be annotated in `sample` folder inside `annotate`.

Step 2. Run the following to perform player detection and pose estimation.
```
cd annotate
python estimate_pose.py sample --from_video
```
Step 3. Run the following to start annotating the on-the-ball players. `sample` indicates the source folder where all video clips are located.
```
python annotation_tool.py sample
```
Step 4. Annotation step-by-step
- Pause the video and locate the On-the-ball player.
- Select On-the-ball player's ID from Player List on the right side of the window. Click “Capture ID”.
- Click “Save Skeleton File” to save the change to CSV annotation file.
---


## Reference

Please consider citing our works if you find it helpful.

```
@inproceedings{Maulana2024Toward,
  title={Toward Soccer Player On-the-ball Action Recognition from Broadcast Match Videos: Data Collection,  Analysis and Findings},
  author={Maulana,  Muhammad Farhan and Ogata,  Kohichi},
  booktitle={Proceedings of the 2024 12th International Conference on Computer and Communications Management},
  pages={79–85},
  year={2024}
}

@inproceedings{Maulana2024Development,
  title={Development of an Annotation Tool for On-the-ball Soccer Player Localization},
  author={Maulana,  Muhammad Farhan and Ogata,  Kohichi},
  booktitle={Record of 2024 Joint Conference of Electrical and Electronics Engineers in Kyushu},
  pages={92},
  year={2024}
}
