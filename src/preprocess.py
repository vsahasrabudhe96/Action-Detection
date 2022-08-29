from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os


def train_test_data(DATA_PATH,actions,no_sequences,sequence_length):
    sequences, labels = [],[]
    label_map = {label:num for num,label in enumerate(actions)}
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window) ## X data (no. of labels* no_sequence,sequence_length,total_keypoints) == (3*20,20,4*33+468*3+21*3 +21*3) == (60,20,1662)
            labels.append(label_map[action]) ## Y data
    
    X = np.array(sequences)
    y = to_categorical(labels).astype('int')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)
    
    return X_train,X_test,y_train,y_test