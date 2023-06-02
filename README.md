# TFG
Final degree code for the project Semantic Segmentation of LUS Retraining a CNN - Ferran Franco 


1 - Dividing_data -> Code that divides the data in k folds, in order to, from one dataset create k different datasets for training and validation. This code wasn't finally used because in order to train with a k fold of 5 folds, the training process is augmented by 5 times. 


2 - Training_Ferran -> Code that was used for training with k-fold datasets. At the end, this code wasn't used. 


3 - Final_Training -> Code used in order to train the network.


4 - Test_Images -> Code used to obtain the different Image & Mask from a .mat file used as test images for network validation. 


5 - Test_Data -> Code used in order to obtain a predicted mask (after training the net) from an input dataset and obtain different values (metrics) related to the comparison of the predicted mask with the original test GT mask. 


6 - Test_Data_B -> Code used in order to obtain a predicted mask (after training the net) from an input dataset and obtain different values (metrics) related to the comparison of the predicted mask with the original test B mask. 


7 - GradCAM -> Code used in order to study, using Grad-CAM tool the different predicted masks (firstly showing the last layer and then some desired layers in between). 
