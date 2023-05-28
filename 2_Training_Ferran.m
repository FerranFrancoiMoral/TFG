clear; clc; close all
% Step 1: Prepare dataset, locate the folders containing the data

%k =1
inputPath_Train1 = 'D:\TFG\input_output\k_1\input_Train1\';
outputPath_Train1 = 'D:\TFG\input_output\k_1\output_Train1\';
inputPath_Val1 = 'D:\TFG\input_output\k_1\input_Val1\';
outputPath_Val1 = 'D:\TFG\input_output\k_1\output_Val1\';

%k =2
inputPath_Train2 = 'D:\TFG\input_output\k_2\input_Train2\';
outputPath_Train2 = 'D:\TFG\input_output\k_2\output_Train2\';
inputPath_Val2 = 'D:\TFG\input_output\k_2\input_Val2\';
outputPath_Val2 = 'D:\TFG\input_output\k_2\output_Val2\';

%k =3
inputPath_Train3 = 'D:\TFG\input_output\k_3\input_Train3\';
outputPath_Train3 = 'D:\TFG\input_output\k_3\output_Train3\';
inputPath_Val3 = 'D:\TFG\input_output\k_3\input_Val3\';
outputPath_Val3 = 'D:\TFG\input_output\k_3\output_Val3\';

%k =4
inputPath_Train4 = 'D:\TFG\input_output\k_4\input_Train4\';
outputPath_Train4 = 'D:\TFG\input_output\k_4\output_Train4\';
inputPath_Val4 = 'D:\TFG\input_output\k_4\input_Val4\';
outputPath_Val4 = 'D:\TFG\input_output\k_4\output_Val4\';

%k =5
inputPath_Train5 = 'D:\TFG\input_output\k_5\input_Train5\';
outputPath_Train5 = 'D:\TFG\input_output\k_5\output_Train5\';
inputPath_Val5 = 'D:\TFG\input_output\k_5\input_Val5\';
outputPath_Val5 = 'D:\TFG\input_output\k_5\output_Val5\';



%% train UNET
%insert size of UNET and determine the number of classes. 
imageSize = [256 256];
numClasses = 2; 

classNames = ["background" "GT"];
labelIDs   = [0,255];

%Create the UNET default net (pre-training)
lgraph = unetLayers(imageSize, numClasses); 


%% Step 1: Get image train file paths for each k fold
imageFiles_Train_1 = dir(fullfile(inputPath_Train1, '*.jpg'));
labelFiles_Train_1 = dir(fullfile(outputPath_Train1, '*.jpg'));

imageFiles_Train_2 = dir(fullfile(inputPath_Train2, '*.jpg'));
labelFiles_Train_2 = dir(fullfile(outputPath_Train2, '*.jpg'));

imageFiles_Train_3 = dir(fullfile(inputPath_Train3, '*.jpg'));
labelFiles_Train_3 = dir(fullfile(outputPath_Train3, '*.jpg'));

imageFiles_Train_4 = dir(fullfile(inputPath_Train4, '*.jpg'));
labelFiles_Train_4 = dir(fullfile(outputPath_Train4, '*.jpg'));

imageFiles_Train_5 = dir(fullfile(inputPath_Train5, '*.jpg'));
labelFiles_Train_5 = dir(fullfile(outputPath_Train5, '*.jpg'));

% Get training image and mask training label file paths K = 1
for idx = 1:numel(imageFiles_Train_1)
    imgFiles_train_1{idx} = fullfile(imageFiles_Train_1(idx).folder, imageFiles_Train_1(idx).name);
    lblFiles_train_1{idx} = fullfile(labelFiles_Train_1(idx).folder, labelFiles_Train_1(idx).name);
end
% Get training image and training label file paths K = 2
for idx = 1:numel(imageFiles_Train_2)
    imgFiles_train_2{idx} = fullfile(imageFiles_Train_2(idx).folder, imageFiles_Train_2(idx).name);
    lblFiles_train_2{idx} = fullfile(labelFiles_Train_2(idx).folder, labelFiles_Train_2(idx).name);
end
% Get training image and training label file paths K = 3
for idx = 1:numel(imageFiles_Train_3)
    imgFiles_train_3{idx} = fullfile(imageFiles_Train_3(idx).folder, imageFiles_Train_3(idx).name);
    lblFiles_train_3{idx} = fullfile(labelFiles_Train_3(idx).folder, labelFiles_Train_3(idx).name);
end
% Get training image and training label file paths K = 4
for idx = 1:numel(imageFiles_Train_4)
    imgFiles_train_4{idx} = fullfile(imageFiles_Train_4(idx).folder, imageFiles_Train_4(idx).name);
    lblFiles_train_4{idx} = fullfile(labelFiles_Train_4(idx).folder, labelFiles_Train_4(idx).name);
end
% Get training image and training label file paths K = 5
for idx = 1:numel(imageFiles_Train_5)
    imgFiles_train_5{idx} = fullfile(imageFiles_Train_5(idx).folder, imageFiles_Train_5(idx).name);
    lblFiles_train_5{idx} = fullfile(labelFiles_Train_5(idx).folder, labelFiles_Train_5(idx).name);
end

%% Step 2: Get image and mask validation file paths for each k fold
imageFiles_Val_1 = dir(fullfile(inputPath_Val1, '*.jpg'));
labelFiles_Val_1 = dir(fullfile(outputPath_Val1, '*.jpg'));

imageFiles_Val_2 = dir(fullfile(inputPath_Val2, '*.jpg'));
labelFiles_Val_2 = dir(fullfile(outputPath_Val2, '*.jpg'));

imageFiles_Val_3 = dir(fullfile(inputPath_Val3, '*.jpg'));
labelFiles_Val_3 = dir(fullfile(outputPath_Val3, '*.jpg'));

imageFiles_Val_4 = dir(fullfile(inputPath_Val4, '*.jpg'));
labelFiles_Val_4 = dir(fullfile(outputPath_Val4, '*.jpg'));

imageFiles_Val_5 = dir(fullfile(inputPath_Val5, '*.jpg'));
labelFiles_Val_5 = dir(fullfile(outputPath_Val5, '*.jpg'));

% Get validation image and validation label file paths K = 1
for idx = 1:numel(imageFiles_Val_1)
    imgFiles_val_1{idx} = fullfile(imageFiles_Val_1(idx).folder, imageFiles_Val_1(idx).name);
    lblFiles_val_1{idx} = fullfile(labelFiles_Val_1(idx).folder, labelFiles_Val_1(idx).name);
end
% Get validation image and validation label file paths K = 2
for idx = 1:numel(imageFiles_Val_2)
    imgFiles_val_2{idx} = fullfile(imageFiles_Val_2(idx).folder, imageFiles_Val_2(idx).name);
    lblFiles_val_2{idx} = fullfile(labelFiles_Val_2(idx).folder, labelFiles_Val_2(idx).name);
end
% Get validation image and validation label file paths K = 3
for idx = 1:numel(imageFiles_Val_3)
    imgFiles_val_3{idx} = fullfile(imageFiles_Val_3(idx).folder, imageFiles_Val_3(idx).name);
    lblFiles_val_3{idx} = fullfile(labelFiles_Val_3(idx).folder, labelFiles_Val_3(idx).name);
end
% Get validation image and validation label file paths K = 4
for idx = 1:numel(imageFiles_Val_4)
    imgFiles_val_4{idx} = fullfile(imageFiles_Val_4(idx).folder, imageFiles_Val_4(idx).name);
    lblFiles_val_4{idx} = fullfile(labelFiles_Val_4(idx).folder, labelFiles_Val_4(idx).name);
end
% Get validation image and validation label file paths K = 5
for idx = 1:numel(imageFiles_Val_5)
    imgFiles_val_5{idx} = fullfile(imageFiles_Val_5(idx).folder, imageFiles_Val_5(idx).name);
    lblFiles_val_5{idx} = fullfile(labelFiles_Val_5(idx).folder, labelFiles_Val_5(idx).name);
end




%%
k = 5;
% Parameters used in the future

resultsDir = 'D:\TFG\input_output\resultsDir\';
subfoldersName = {'SemanticsegOutput_1', 'SemanticsegOutput_2',...
    'SemanticsegOutput_3','SemanticsegOutput_4',...
    'SemanticsegOutput_5'};

variableTypes = {'double', 'double', 'double', 'double', 'double'};
complete_metrics= table('Size',[0,5],'VariableTypes',variableTypes,'VariableNames',{'GlobalAccuracy',...
    'MeanAccuracy','MeanIoU','WeightedIoU','MeanBFScore'});

for netIDX = 1:k

    % Step 3:  Create Datastore
    if netIDX == 1
        %k=1
        imds_train = imageDatastore(inputPath_Train1);
        pxds_train = pixelLabelDatastore(lblFiles_train_1,classNames,labelIDs);
        lista = natsortfiles(pxds_train.Files);
        pxds_train = pixelLabelDatastore(lista,classNames,labelIDs);
        imds_val = imageDatastore(inputPath_Val1);
        pxds_val = pixelLabelDatastore(lblFiles_val_1,classNames,labelIDs);
        lista = natsortfiles(pxds_val.Files);
        pxds_val = pixelLabelDatastore(lista,classNames,labelIDs);

    elseif netIDX == 2
        %k=2
        imds_train = imageDatastore(inputPath_Train2);
        pxds_train = pixelLabelDatastore(lblFiles_train_2,classNames,labelIDs);
        lista = natsortfiles(pxds_train.Files);
        pxds_train = pixelLabelDatastore(lista,classNames,labelIDs);
        imds_val = imageDatastore(inputPath_Val2);
        pxds_val = pixelLabelDatastore(lblFiles_val_2,classNames,labelIDs);
        lista = natsortfiles(pxds_val.Files);
        pxds_val = pixelLabelDatastore(lista,classNames,labelIDs);


    elseif netIDX == 3
        %k=3
        imds_train = imageDatastore(inputPath_Train3);
        pxds_train = pixelLabelDatastore(lblFiles_train_3,classNames,labelIDs);
        lista = natsortfiles(pxds_train.Files);
        pxds_train = pixelLabelDatastore(lista,classNames,labelIDs);
        imds_val = imageDatastore(inputPath_Val3);
        pxds_val = pixelLabelDatastore(lblFiles_val_3,classNames,labelIDs);
        lista = natsortfiles(pxds_val.Files);
        pxds_val = pixelLabelDatastore(lista,classNames,labelIDs);

    elseif netIDX == 4
        %k=4
        imds_train = imageDatastore(inputPath_Train4);
        pxds_train = pixelLabelDatastore(lblFiles_train_4,classNames,labelIDs);
        lista = natsortfiles(pxds_train.Files);
        pxds_train = pixelLabelDatastore(lista,classNames,labelIDs);
        imds_val = imageDatastore(inputPath_Val4);
        pxds_val = pixelLabelDatastore(lblFiles_val_4,classNames,labelIDs);
        lista = natsortfiles(pxds_val.Files);
        pxds_val = pixelLabelDatastore(lista,classNames,labelIDs);

    elseif netIDX == 5
        %k=5
        imds_train = imageDatastore(inputPath_Train5);
        pxds_train = pixelLabelDatastore(lblFiles_train_5,classNames,labelIDs);
        lista = natsortfiles(pxds_train.Files);
        pxds_train = pixelLabelDatastore(lista,classNames,labelIDs);
        imds_val = imageDatastore(inputPath_Val5);
        pxds_val = pixelLabelDatastore(lblFiles_val_5,classNames,labelIDs);
        lista = natsortfiles(pxds_val.Files);
        pxds_val = pixelLabelDatastore(lista,classNames,labelIDs);

    end
    %Step 4: Create combined datastore
    ds = combine(imds_train,pxds_train);
    dsval = combine(imds_val, pxds_val);
    
    
    %% Step 5: Set training options with the hyperparameters.


    options = trainingOptions("adam", ...
        Shuffle="every-epoch", ...
        InitialLearnRate=0.001, ...
        MaxEpochs=10, ...
        MiniBatchSize=8, ...
        ValidationData=dsval, ...
        ValidationFrequency=10, ...
        Verbose=0, ...
        ExecutionEnvironment="gpu",...
        Plots="training-progress", ...
        OutputNetwork="best-validation-loss");

    %Retrain the model
    fprintf('Training fold %d/%d...\n', netIDX, k);
    net = trainNetwork(ds,lgraph,options);

    %To divide for each k the result files on different folders
    subfolderName = subfoldersName{netIDX};
    folderPath = fullfile(resultsDir, subfolderName);
    mkdir(folderPath);

    %% Test the network creating a predicted mask

    fprintf('Validation metrics for fold %d/%d:\n', netIDX, k);
    pxdsPred = semanticseg(imds_val, net,...
        'MiniBatchSize', 8,...
        'WriteLocation',folderPath,...
        'Classes',classNames,...
        'OutputType', 'categorical',...
        'ExecutionEnvironment',"gpu");


    metrics = evaluateSemanticSegmentation(pxdsPred,pxds_val);
    complete_metrics = vertcat(complete_metrics, metrics.DataSetMetrics);

end

