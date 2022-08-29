# STEPS TO FOLLOW FOR PERFORMING ACTION DETECTION

### 1) Import the libraries
- Numpy
- Pandas
- Matplotlib
- Opencv(cv2)
- Keras
- Tensorflow
- Mediapipe

### 2) Keypots using MP Holistics
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
- We want to collect a sequence of frames for real time prediction
- We initialize the action which we require to capture data for
- We initialize the amount of videos we want to collect for each action (number  of sequence)
- We initialize the frames for each of the videos (sequence length)
### 5) Collect Keypoint Values for Training and Testing
- Create a separate python script for data collection so that we don't perform the action again and again
- run `python datacollection.py` for data collection process
- In this file we will inpute the number of sequences, the data path and length of the sequences that we want capture via terminal arg parse
- Introduce a time out /pause to capture data effectively
- Store the keypoints in specific data folder 
### 6) Preprocess Data and Create Labels and Features
- Create a label map to map the actions to integer values
- Accumulate all the keypoints of the sequence length for a particular sequence
- Accumulate all the keypoints for all such sequences into the another array which will form our feature variable X 
- Shape of X will be (no. of labels* no_sequence,sequence_length,total_keypoints)

- Create labels by accumulating all the action values of label map for all sequence

- Convert the labels into categorical using keras
- Split using scikit learn

### 7) Build and Train LSTM Neural Network
- We are using LSTM model over here, as we are going to collect sequential data for training
- Training the model for 50 epochs, Adam optimizer
- run `python model.py` for training the model

### 8) Make Predictions
- Pass the test data points through the model for preddiction
### 9) Save Weights
- Saving the weight in a folder

### 10) Test in Real Time



Code Reference: 
Nicholas Renotte
https://github.com/nicknochnack/ActionDetectionforSignLanguage

