clear; clc; close all

% Prepare dataset

inputPath_Train = 'D:\TFG\input_output\test\training\input_test\';
outputPath_Train = 'D:\TFG\input_output\test\training\output_test\';
inputPath_Val = 'D:\TFG\input_output\test\validation\input_val\';
outputPath_Val = 'D:\TFG\input_output\test\validation\output_val\';

inputPath_Test = 'D:\TFG\input_output\test\validation\input_val\';
outputPath_Test = 'D:\TFG\input_output\test\validation\output_val\';


%% create UNET
%Define the image size and the number of classes
imageSize = [256 256];
numClasses = 2;

classNames = ["background" "GT"];
labelIDs   = [0,255];

lgraph = unetLayers(imageSize, numClasses);


%% Step 1: Get image train file paths 
imageFiles_Train = dir(fullfile(inputPath_Train, '*.jpg'));
labelFiles_Train = dir(fullfile(outputPath_Train, '*.jpg'));

% Get training image and mask training label 
for idx = 1:numel(imageFiles_Train)
    imgFiles_train{idx} = fullfile(imageFiles_Train(idx).folder, imageFiles_Train(idx).name);
    lblFiles_train{idx} = fullfile(labelFiles_Train(idx).folder, labelFiles_Train(idx).name);
end


%% Step 2: Get image and mask validation file paths 
imageFiles_Val = dir(fullfile(inputPath_Val, '*.jpg'));
labelFiles_Val = dir(fullfile(outputPath_Val, '*.jpg'));

% Get validation image and validation label file paths
for idx = 1:numel(imageFiles_Val)
    imgFiles_val{idx} = fullfile(imageFiles_Val(idx).folder, imageFiles_Val(idx).name);
    lblFiles_val{idx} = fullfile(labelFiles_Val(idx).folder, labelFiles_Val(idx).name);
end

%% Step 3: Get image and mask test file paths 
imageFiles_Test = dir(fullfile(inputPath_Test, '*.jpg'));
labelFiles_Test = dir(fullfile(outputPath_Test, '*.jpg'));

% Get test image and test label file paths
for idx = 1:numel(imageFiles_Test)
    imgFiles_test{idx} = fullfile(imageFiles_Test(idx).folder, imageFiles_Test(idx).name);
    lblFiles_test{idx} = fullfile(labelFiles_Test(idx).folder, labelFiles_Test(idx).name);
end


% Create Datastore

%Train datastore and pixellabeldatastore
imds_train = imageDatastore(inputPath_Train);
pxds_train = pixelLabelDatastore(lblFiles_train,classNames,labelIDs);
list = natsortfiles(pxds_train.Files);
pxds_train = pixelLabelDatastore(list,classNames,labelIDs);

%Validation datastore and pixellabeldatastore
imds_val = imageDatastore(inputPath_Val);
pxds_val = pixelLabelDatastore(lblFiles_val,classNames,labelIDs);
list= natsortfiles(pxds_val.Files);
pxds_val = pixelLabelDatastore(list,classNames,labelIDs);

%Test datastore and pixellabeldatastore
imds_test = imageDatastore(inputPath_Test);
pxds_test = pixelLabelDatastore(lblFiles_test,classNames,labelIDs);
list = natsortfiles(pxds_test.Files);
pxds_test = pixelLabelDatastore(list,classNames,labelIDs);


%Step 4: Create combined datastore
ds = combine(imds_train,pxds_train);
dsval = combine(imds_val, pxds_val);

%% Step 5: Set training options.

% Here is where the hyperparameters have been modified in different
% trainings. 
options = trainingOptions("adam", ...
    'Shuffle',"every-epoch", ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',32, ...
    'ValidationData',dsval, ...
    'ValidationFrequency',10, ...
    'Verbose',0, ...
    'ExecutionEnvironment',"gpu",...
    'Plots',"training-progress", ...
    'OutputNetwork',"best-validation-loss");

%Train the model
fprintf('Training net...');
[net,info] = trainNetwork(ds,lgraph,options);


% Net information

net_info = table(mean(info.TrainingLoss),mean(info.TrainingAccuracy),...
    mean(info.ValidationLoss),mean(info.ValidationAccuracy),...
    'VariableNames', {'TrainingAccuracy', 'TrainingLoss',...
    'ValidationAccuracy', 'ValidationLoss'});












