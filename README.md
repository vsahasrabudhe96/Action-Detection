# STEPS TO FOLLOW FOR PERFORMING ACTION DETECTION

### 1) Import the libraries
- Numpy
- Pandas
- Matplotlib
- Opencv(cv2)
- Keras
- Tensorflow
- Mediapipe

### 2) Keypoints using MP Holistics
- initialize the Mediapipe model
- Make sure to change the frame/image from  BGR to RGB as mediapipe takes RGB input
- change the image to unwriteable
- process the results by passing image through the model
- change the image to writeable
- reconvert the image from RGB to BGR
- instantiate model before video capture
- call the transformation and drawing function inside the video feed code
	
### 3) Extracting Keypoint Values
- Accumulate all the keypoints in an array
- Error handling will be done by assigning a zero value to a non existing item (e.g if right hand missing, then all the keypoints will be denoted by an array of zeroes)
- Flattening the data to make it compatible to input format of LSTM


### 4) Setup Folders for Collection

### 5) Collect Keypoint Values for Training and Testing

### 6) Preprocess Data and Create Labels and Features


### 7) Build and Train LSTM Neural Network
- We are using LSTM model over here, as we are going to collect sequential data for training

### 8) Make Predictions


### 9) Save Weights


### 10) Evaluation using Confusion Matrix and Accuracy



Code Reference: 
Nicholas Renotte
https://github.com/nicknochnack/ActionDetectionforSignLanguage

