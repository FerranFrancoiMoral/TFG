%% GRAD-CAM 

clc; close all; clear all; 

%Load the desired trained net.
net_name = "myTrainedUnet_0.01_4_4_adam.mat";

Trained_net = load(net_name);
net = Trained_net.net;

% Define the number of classes and which ones there are
classNames = ["background" "GT"];

% function used to analyze the network and understand which are the layers
% that have an special interest in being analyzed. 

analyzeNetwork(net);


% Typically this is a ReLU layer which takes the output of a convolutional 
% layer at the end of the network.

featureLayer = "Decoder-Stage-4-ReLU-2";

%For semantic segmentation problems, the reduction layer is usually the 
% softmax layer.

reductionLayer = "Softmax-Layer";

%Loading one image to study

img_original = imread('D:\TFG\our_simulated_data_testData\input_test\01325.png');

% Load the corresponding mask
mask_original = imread('D:\TFG\our_simulated_data_testData\output_test\01325.png'); 

% Load the predicted mask created on codes 5_Test_Data / 6_Test_Data_B
img = imread('D:\TFG\input_output\resultsDir\myTrainedUnet_0.01_4_4_sgdm\pixelLabel_1325_01325.png');
inputSize = net.Layers(1).InputSize(1:2);
img = imresize(img,inputSize);


%% GRADCAM

% Here the grad-CAM is performed with the different options
gradCAMMap = gradCAM(net,img,classNames, ...
    ReductionLayer=reductionLayer, ...
    FeatureLayer=featureLayer);


%% Plotting the results from the last layer
figure
subplot(2,2,1)
imshow(img_original)
title("Input Image")
subplot(2,2,2)
imshow(mask_original)
title("Output Image")
subplot(2,2,3)
imshow(img)
hold on
imagesc(gradCAMMap(:,:,1))
title("Grad-CAM: " + classNames(1))
colormap jet
subplot(2,2,4)
imshow(img)
hold on
imagesc(gradCAMMap(:,:,2))
title("Grad-CAM: " + classNames(2))
colormap jet

%% GRAD-CAM from different high interest layers
numClasses = length(classNames);

% As a output layer we will choose each output layer from each weight shown
% on the plot resulting of the analyzeNetwork(net) function. 

layers = [string(net.Layers(41,1).Name), string(net.Layers(51,1).Name),...
    string(net.Layers(61,1).Name), string(net.Layers(71,1).Name)]; 

numLayers = length(layers);
gradCAMMaps = [];

%Here the grad-CAM is performed for each one of the desired layers
for i = 1:numLayers
    gradCAMMaps(:,:,:,i) = gradCAM(net,img,classNames, ...
        ReductionLayer=reductionLayer, ...
        FeatureLayer=layers(i));
end

%% Plotting the results from the desired layers
figure;
idx = 1;
for i=1:numLayers
    for j=1:numClasses
        subplot(numLayers,numClasses,idx)
        imshow(img)
        hold on
        imagesc(gradCAMMaps(:,:,j,i),AlphaData=0.5)
        title(sprintf("%s (%s)",classNames(j),layers(i)), ...
            Interpreter="none")
        colormap jet
        idx = idx + 1;
    end
end

