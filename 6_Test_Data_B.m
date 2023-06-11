clc; clear all; close all; 

%Insert the folders with the test images and masks obtained on
% 4_Test_Images (with the modification of just using B-mask on this case)

inputPath_Test = 'D:\TFG\our_simulated_data_testData\input_test\';
outputPath_Test = 'D:\TFG\our_simulated_data_testData\output_B_test\';

% Define the number of classes and which ones there are

numClasses = 2;

classNames = ["background" "B"];
labelIDs   = [0 255];

%% Step 3: Get image and mask test file paths 
imageFiles_Test = dir(fullfile(inputPath_Test, '*.png'));
labelFiles_Test = dir(fullfile(outputPath_Test, '*.png'));

% Get test image and test label file paths
for idx = 1:numel(imageFiles_Test)
    imgFiles_test{idx} = fullfile(imageFiles_Test(idx).folder, imageFiles_Test(idx).name);
    lblFiles_test{idx} = fullfile(labelFiles_Test(idx).folder, labelFiles_Test(idx).name);
end

% Create test image and pixelLabel Datastores
imds_test = imageDatastore(inputPath_Test);
pxds_test = pixelLabelDatastore(lblFiles_test,classNames,labelIDs);
list = natsortfiles(pxds_test.Files);
pxds_test = pixelLabelDatastore(list,classNames,labelIDs);

%Load the pretrained network
net_name = "myTrainedUnet_0.01_4_4_sgdm_onlyBL.mat";

Trained_net = load(net_name);
net = Trained_net.net;

resultsDir = 'D:\TFG\input_output\resultsDir\';

%% Test network creating a predicted B mask and compare it with the output
%% original test mask. 

fprintf('Validation metrics for the net');
pxdsPred = semanticseg(imds_test, net,...
    'MiniBatchSize', 32,...
    'WriteLocation',resultsDir,...
    'Classes',classNames,...
    'OutputType', 'categorical',...
    'ExecutionEnvironment',"gpu");

metrics = evaluateSemanticSegmentation(pxdsPred,pxds_test);
complete_metrics = table(metrics.DataSetMetrics);



