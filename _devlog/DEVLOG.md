# Devlog
### Superpoint paper!! (Feb 13, 2023)
Looked at Superpoint paper (https://arxiv.org/pdf/1712.07629v4.pdf) to see if it is suited for
board detection. <br>
Key take away:
- Upscale output to (1, H, W) from features.
  - This would be key points in white pixels
- Self-Supervise by using 3d generated chess models and keypoints (use Blender)
  - This can be used to create large dataset for **Piece Detector** too

### Massive Changes (Feb 12, 2023)
#### Why a Change of plans
The initial solution was to predict four corners of the board using
simple conv architecture such as Densenet or resnet, warp it and 
then send it into object detection to detect pieces. By warping the image
by the corners of the board, I can estimate location of all squares simply by
using a grid system.

Best case warp scenario:<br>
<img alt="perfect_before_warp.jpg" src="images/23feb12/perfect_before_warp.jpg" width="300px"/><img src="images/23feb12/perfect_warp_example.jpg" width="300px"/>

Worst case warp scenario:<br>
<img src="images/23feb12/bad_before_warp.jpg" width="300px"/> <img src="images/23feb12/bad_warp_example.jpg" width="300px"/>

The problems with this solution, after the warp:
- the images were lower in resolution.
- the images were too cropped cutting some pieces.
- pieces were also skewed too much, depending on image angle.
This made it very difficult for **piece detector** to detect pieces.

#### New Solution
The job of the **board detector** should be to detect every square. I tried using hough transform from
the cv2 library to detect the squares on the board, but unfortunately the detections are not very accurate.
The detections tend to go outside the board and even on pieces and players' clothing when resolution is high.

Hough transform on low res (300px):<br>
<img src="images/23feb12/detectedLines.jpg" width="300px"/>

Hough transform on high res (800px):<br>
<img src="images/23feb12/detectedLines_hires.jpg" width="300px"/>


Then the **piece detector** would work on the same image to detect bbox for each piece.
putting them together, we can get accurate detections without compromising piece detector.<br>
<img src="images/23feb12/new_idea.jpg" width="300px"/>

One small problem with this solution could be that, sometimes piece bboxs can be over 
two squares. This can potentially be solved by looking at only bottom half of bbox because
the bottom of the piece will always be in the right square.
<br><br>

#### Roboflow to VIA
It was tedious and slow to upload images to roboflow to annotate. The solution
was to find a local annotation software to keep all images in one place. Looked at
label studio, but found VIA (VGG image annotator). Which was easy to use and did not
need any pip installations.
<br><br>

#### device("mps") ??
Learned about MPS in pytorch to speed up training on Apple Silicon. Also found
an issue with torch.topk only works for k<=16 when device="mps" and 
torch.topk is used by fasterrcnn for NMS (non-max-suppression). The issue is 
currently being worked on by pytorch contributors. So there is not much I can 
do other than, wait.

### Overfitting Board Detector Issue (Feb 2, 2023)
Focused on overfitting board_detector data. After [0.3.1] (changes to
board detector dataset), there was a major issue that was found. Overfitting only
happened to particularly same few images while rest of the images were completely off.

After a ton of debugging, from ensuring the data is being shuffled, to
checking individual images and figuring out exact overfitting images. The problem
was simply with incorrect length function (__len__) in board dataset module.

### Dealing with GitHub and large files (Jan 1, 2023)
The saved files from training piece detector were too big, therefore
had issues with GitHub accepting such large files. Used force push to push changes 
since local git and GitHub did not have same branch. Had to figure out some 
trickery to finally sync local git and GitHub.

### Massive fix to the piece detector model (May 11, 2023)
I realized that FasterRCNN does not normalize the bbox predictions. Where as, my dataset 
had normalized the target bbox for the FasterRCNN. This has caused the FasterRCNN model
to take an extremely large iterations to see noticeable gains. Therefore, after **fixing the
dataset by removing the normalization for bboxes**, the model performed a lot better with very
few iterations.<br>

<img src="images/23may11/piece_detector_1500_steps.jpg" width="800px"/><br>

_left: target bboxes, right: prediction bboxes_<br><br>
The was successfully able to over-fit to a small batch size. which indicates the model 
is going to work for a larger dataset.

#### New implementation: Tensorboard
Along with the major fix, I have implemented tensorboard to visualize and track the losses to better
evaluate the model. I have **begun hyperparameter search and found that batch size of 2 and learning rate of
1e-4 worked best for this small dataset**. I plan to continue to further search for better hyperparameters.
<br>
<img src="images/23may11/losses.jpg" height="300px"/><br>
_loss_rpn_box_reg (dark red): loss of the predicted box vs ground truth box_<br>
_loss_box_red (pink): loss of predicted location of bbox vs ground truth box location_<br>
_loss_objectiveness (light blue): loss of if there is an object in bbox_<br>
_loss_classifier (blue): classification loss of piece in bbox_<br><br>

### Synthetic Data Generation (May 19, 2023)

#### Plan
The motivation behind synthetic data generation is to avoid 
the tedious task of collecting and labeling chess images by hand, 
which can take weeks to collect and label just 100 images. My plan was
to generate synthetic data using Blender, since it allows for python scripting.

#### Action
Although the idea was simple, the implementation was difficult. Fortunately, I
had a fair bit of experience using Blender which made it a little easier.

###### Step 1. Create a realistic scene
I had to search and download 3D piece models, I had to learn about creating textures, using nodes, 
principled BSDF, different lighting and UV unwrapping.

###### Step 2. Write code to create random scenes
This involved writing python code to place camera in different positions while
still looking at the board. Setting up random positions on the board
using fen as input. There are still a lot more randomness that could take place
such as different piece types, board color, table material, lighting change, and
various environment props.

###### Step 3. Write code to create annotations
This was difficult to come up with entirely from my own code, so I have used
help from the community. Below are the links:

https://github.com/georg-wolflein/chesscog/blob/master/scripts/synthesize_data.py#L195 <br>
https://blender.stackexchange.com/a/158236 <br>
https://stackoverflow.com/questions/63753960 <br>

#### output example:
<img src="images/23may19/example.jpg" width="500">

#### output example with annotations:
<img src="images/23may19/output.png" width="500">

Clearly, the annotations are far more accurate than humans, and
the variation of the image can be limitless. It really comes down
to how realistic I can render the scene.







