# Changelog
## [1.0.1] - 2023-
### Changes
- resized _devlog images to reduce file sizes
### Added
- areas into the target of Piece Detector
- 
## [1.0.0] - 2023-02-12
### Changes
- switched from roboflow annotator to VIA
- renamed folders to because ultralytics yolo package names clashed 
- training piece detector without warping

### Added
- CHANGELOG.md
- _devlog folder to blog everything
- resize.py to resize images in a folder
- New images data with higher resolution
- git is going to ignore data folder, due to GitHub size restrictions
- hough.py added to test board detections without CNN

## [0.4.3] - 2023-02-2
### Summary
Focused on overfitting board_detector data. After [0.3.1] (changes to
board detector dataset), there was a major issue that was found. Overfitting only
happened to particularly same few images while rest of the images were completely off.

After a ton of debugging, from ensuring the data is being shuffled, to
checking individual images and figuring out exact overfitting images. The problem
was simply with incorrect length function (__len__) in board dataset module.

### Changes
- correcting __len__ function in board dataset class.

## [0.4.2] - 2023-01-30
### Summary
used force push to push changes since local git and github did
not have same branch. Had to figure out some trickery to finally sync
local git and github.

## [0.4.1] - 2023-01-30
### Summary
updated .gitignore file to ignore checkpoints because of issues with 
pushing large trained files from Piece Detector.

### Changes
- updated .gitignore file to ignore checkpoints

## [0.4.0] - 2023-01-29
### Summary
converted from labelbox to roboflow for ease of use and 
coco.json annotations output.

Learned about mixed precision training and incorporated it to both
piece and board trainers to speed up training.
### Changes
- dataloader/data.py
  - converted from labelbox annotations to roboflow annotator
### Added
- testing.py file
  - show_board_detector_results function to visualize output from board detector
  - show_piece_detector_results function to visualize output from piece detector
- Mixed precision training to ensure faster training on GPU

## [0.3.0] - 2023-01-22
### Summary
Trainer added for Piece Detector using Adam optimizer and learning rate of 3e-4.
the loss is a summation of losses dict output from fasterrcnn().

## [0.2.1] - 2023-01-17
### Summary
Clean up code and organized folders along with clear documentation of
functions

## [0.2.0] - 2023-01-17
### Summary
created Piece Dataset module to prepare image data for training

### Added
- Piece Detector Dataset

## [0.1.0] - 2023-01-14
### Summary
Renamed skew to warp because warp sounds better.
Importantly vectorized warp function using Kornia library.

### Added
- new file warp.py using kornia libray

### Removed
- skew.py 

## [0.0.4] - 2023-01-14
### Summary
readme file updated

## [0.0.3] - 2023-01-14
### Summary
small bug fix with utils __init__.py file

## [0.0.2] - 2023-01-14
### Summary
Organized code by using best practices and documentation along with .gitignore.

## [0.0.1] - 2023-01-14
### Summary
Created Board Detector model with Densenet201() with 
8 outputs for detecting each corner of the board. Using 11 images of
chessboards taken with an iPhone 13 Pro camera with square ratio, the
model was trained with MSELoss() with 3e-4 learning rate. The goal was to
overfit the 11 images data to ensure good model performance on large data.

Skew function added to be used after detecting the 4 corners. The skew function
will skew the image to fit and fill the entire board perfectly in the center of
the new image. Which will then be sent into Piece Detector to detect pieces

Created Piece Detector model which was just fasterrcnn.

### Added
- Board detector model created with basic Densenet201() with 
8 outputs for detecting each corner of the board.
- Trainer for the model is created with MSELoss()
- Dataset of 11 images
- Created piece detector model
- skew function 