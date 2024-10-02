# Sign-Language-Detection-System-To-sentence
American Sign Language (ASL) is the visual-general language used by deaf people in the United
States and parts of Canada. There is no universal sign language, different sign languages are used
in different countries or regions. No person or committee invented ASL the exact beginning of
ASL is not clear, but some suggest that it arose more than 200 years ago from the intermixing of
local sign languages and French sign language. The movements of hands and face express ASL It
is the primary language of many North Americans who are deaf and hard of hearing, and it is used
by many hearing people as well. ASL has a large area ofscope. Fingerspelling is part of ASL and is
used to spell out English words. In the figure-spelled alphabet, each letter corresponds to a distinct
handshape. 
# How to run this code 
  in first download this repository to your local computer then after setup python with the required version 

** step 2** : run file name imagecapture.py and start capturing hand landmark then image i saved on directory 
**step 2** : after capturing dataset you can train data by using file name CNNmodel.py and model is saved which is further for data prediction
**step 3:** run file name final_pred.py which contains GUI and then place your hand then it show landmark on white image and image is predicted and show the suggestion also and click on speak for sound and clear for clearing all the text and sentence

# This screenshot show the picture of landmark that is capture to train data

![Screenshot 2024-09-30 152428](https://github.com/user-attachments/assets/680b53f3-4095-4f06-bccc-1b2f8c535ceb)

# This hand mark show the point of whist and five finger which is used in code for high accuracy of data 

![Screenshot 2024-09-30 071357](https://github.com/user-attachments/assets/6c8b63c9-c436-4837-8d19-f714897b888c)

# This is screenshot of model training :
![Screenshot (1)](https://github.com/user-attachments/assets/b696538c-ddc0-47a4-8152-9c36f077af56)

![Screenshot 2024-09-28 203025](https://github.com/user-attachments/assets/a2adea33-b224-49f6-9dfc-b524f4fe99a6)



# This is the demo of this project:


https://github.com/user-attachments/assets/e1a2248a-4068-44ae-a771-3cd0e9eac871

# implementation of CNN

**First Convolutional Block:** A Conv2D layer with 32 filters (3x3), using ReLU, takes 400x400
RGB images, followed by MaxPooling to downsample.
**Second Convolutional Block:** Adds a Conv2D layer with 64 filters and MaxPooling to capture
more complex features.
**Third Convolutional Block:** Another Conv2D layer with 128 filters and MaxPooling for deeper
feature learning.
**Flattening Layer:** Flattens the output from the convolutional layers into a 1D vector for the dense
layer.
**Dense Layer:** A fully connected layer with 512 neurons using ReLU to combine learned features.
**Dropout Layer:** A Dropout layer (50%) to reduce overfitting.
**Output Layer:** A Dense layer with softmax activation for classifying 26 classes (A-Z). Model
**Summary:** Prints the architecture details, including layers and parameters.





