# Chess Board Detection
The goal of this project is to have users take a picture of any tournament chessboard position and convert it into a digital position ready to be analyized by an engine. This can also be extened to recording realtime games without the use of expensive hardware to track the position.

### TODO:
- create piece detector dataset
- write training function for piece detector

### Labeling Process:
- Take picture of a board
    - image taken has a square ratio
    - image is of size 320x320 or higher
- Label the board using labelbox
    - Board labelers need to ensure that order of points should be (a8, h8, h1, a1)
        - labeling this way warps the board in the same orientation everytime.
        - during training the model will learn the order which outputs correct orientation during inferance.