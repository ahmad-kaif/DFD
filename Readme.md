# DeepFake-Detect

## Description
Trying to make a deepfake detection model

### Deepfake Datasets


- [DeepFake-TIMIT](https://www.idiap.ch/dataset/deepfaketimit)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Google Deep Fake Detection (DFD)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
- [Celeb-DF](https://github.com/danmohaha/celeb-deepfakeforensics)
- [Facebook Deepfake Detection Challenge (DFDC)](https://ai.facebook.com/datasets/dfdc/)

<p align="center"><img alt="" src="https://github.com/aaronchong888/DeepFake-Detect/blob/master/img/sample_dataset.png" width="80%"></p>


## Getting Started

### Prerequisites

- Python 3
- Keras
- TensorFlow
- EfficientNet for TensorFlow Keras
- OpenCV on Wheels
- MTCNN

### Installation

```
pip install -r requirements.txt
```

### Usage

#### Step 0 - Convert video frames to individual images

```
python3 01-convert-video-to-image.py
```

Extract all the video frames from the acquired deepfake datasets above, saving them as individual images for further processing. In order to cater for different video qualities and to optimize for the image processing performance, the following image resizing strategies were implemented:

- 2x resize for videos with width less than 300 pixels
- 1x resize for videos with width between 300 and 1000 pixels
- 0.5x resize for videos with width between 1000 and 1900 pixels
- 0.33x resize for videos with width greater than 1900 pixels

#### Step 1 - Extract faces from the deepfake images with MTCNN

```
python3 02-crop-faces-with-mtcnn.py
```

Further process the frame images to crop out the facial parts in order to allow the neural network to focus on capturing the facial manipulation artifacts. In cases where there are more than one subject appearing in the same video frame, each detection result is saved separately to provide better variety for the training dataset.

- The pre-trained MTCNN model used is coming from this GitHub repo: https://github.com/ipazc/mtcnn
- Added 30% margins from each side of the detected face bounding box
- Used 95% as the confidence threshold to capture the face images



#### Step 2 - Balance and split datasets into various folders

```
python3 03-prepare-fake-real-dataset.py
```

As we observed that the number of fakes are much larger than the number of real faces (due to the fact that one real video is usually used for creating multiple deepfakes), we need to perform a down-sampling on the fake dataset based on the number of real crops, in order to tackle for possible class imbalance issues during the training phase. 

We also need to split the dataset into training, validation and testing sets (for example, in the ratio of 80:10:10) as the final step in the data preparation phase.

#### Step 3 - Model training

```
python3 04-train-cnn.py
```

EfficientNet is used as the backbone for the development work. Given that most of the deepfake videos are synthesized using a frame-by-frame approach, we have formulated the deepfake detection task as a binary classification problem such that it would be generally applicable to both video and image contents.

In this code sample, we have adapted the EfficientNet B0 model in several ways: The top input layer is replaced by an input size of 128x128 with a depth of 3, and the last convolutional output from B0 is fed to a global max pooling layer. In addition, 2 additional fully connected layers have been introduced with ReLU activations, followed by a final output layer with Sigmoid activation to serve as a binary classifier. 

Thus, given a colored square image as the network input, we would expect the model to compute an output between 0 and 1 that indicates the probability of the input image being either deepfake (0) or pristine (1).





