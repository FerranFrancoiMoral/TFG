%% Dividing data into train  and validation
%https://es.mathworks.com/matlabcentral/answers/281222-how-to-assign-my-defined-training-set-validation-set-and-test-set-for-training-a-neural-net-in-neur
clc
clear all
close all
% Load the images and mask files (located on .mat files)
fds = fileDatastore(fullfile('D:\TFG\matfiles\OrganizedData\organizedData'),"ReadFcn",@load,"FileExtensions",".mat");
k = 5;
num_files = numel(fds.Files);

%Obtain the name where the data will be found at the beginning and for each
% fold

inputPath='D:\TFG\input_output\input\';
outputPath='D:\TFG\input_output\output\';

%k = 1
inputPath_Train1 = 'D:\TFG\input_output\k_1\input_Train1\';
outputPath_Train1 = 'D:\TFG\input_output\k_1\output_Train1\';
inputPath_Val1 = 'D:\TFG\input_output\k_1\input_Val1\';
outputPath_Val1 = 'D:\TFG\input_output\k_1\output_Val1\';

%k = 2
inputPath_Train2 = 'D:\TFG\input_output\k_2\input_Train2\';
outputPath_Train2 = 'D:\TFG\input_output\k_2\output_Train2\';
inputPath_Val2 = 'D:\TFG\input_output\k_2\input_Val2\';
outputPath_Val2 = 'D:\TFG\input_output\k_2\output_Val2\';

%k = 3
inputPath_Train3 = 'D:\TFG\input_output\k_3\input_Train3\';
outputPath_Train3 = 'D:\TFG\input_output\k_3\output_Train3\';
inputPath_Val3 = 'D:\TFG\input_output\k_3\input_Val3\';
outputPath_Val3 = 'D:\TFG\input_output\k_3\output_Val3\';

%k = 4
inputPath_Train4 = 'D:\TFG\input_output\k_4\input_Train4\';
outputPath_Train4 = 'D:\TFG\input_output\k_4\output_Train4\';
inputPath_Val4 = 'D:\TFG\input_output\k_4\input_Val4\';
outputPath_Val4 = 'D:\TFG\input_output\k_4\output_Val4\';

%k = 5
inputPath_Train5 = 'D:\TFG\input_output\k_5\input_Train5\';
outputPath_Train5 = 'D:\TFG\input_output\k_5\output_Train5\';
inputPath_Val5 = 'D:\TFG\input_output\k_5\input_Val5\';
outputPath_Val5 = 'D:\TFG\input_output\k_5\output_Val5\';

%% Load the images and masks from the .mat files and store them on the
%the folder located in inputPath (image) and outputPath (mask)

for idx = 1:num_files
    file = load(fds.Files{idx,1});
    in = imresize(file.lusData.imgInfo.img,[256 256]);
    name=fds.Files{idx,1};
    name=name(end-9:end-4);
    imgName=append(inputPath,name,'.jpg');
    imwrite(in , imgName);
    imageFilenames{idx,1}=imgName;

    out = imresize(file.lusData.maskInfo.GT,[256 256]);
    maskName=append(outputPath,name,'.jpg');
    imwrite(out , maskName);
    maskFilenames{idx,1}=maskName;
end

%% Create a datastore for the images and another one for the Masks

inputIMG = imageDatastore(inputPath,"FileExtensions",'.jpg');
outputIMG = imageDatastore(outputPath,"FileExtensions",'.jpg');

% create a k-fold partition
cv = cvpartition(num_files,'Kfold',k);

%% In order to divide the data between train and validation

%Define the class names and their associated label IDs.
classNames = ["background","GT"];
labelIDs   = [0,255];

%Create a pixelLabelDatastore object to store the ground truth pixel labels
% for the training images.
pxds = pixelLabelDatastore(outputIMG.Files,classNames,labelIDs);
list = natsortfiles(pxds.Files);
pxds = pixelLabelDatastore(list,classNames,labelIDs);


%% Loop through each fold in order to create train and validation sets
for i = 1:k
    train_idx = cv.training(i);
    val_idx = cv.test(i);
    % Create a subset of the datastore for the current fold
    train_Image_Datastore{i,:} = subset(inputIMG, train_idx);
    train_Mask_Datastore{i,:} = subset(pxds, train_idx);

    %Create an imageDatastore object to store the training images and masks
    %for each k-fold

    %For fold 1
    if i==1
        imds_train_1 = train_Image_Datastore{i,:};
        pxl_train_1 = train_Mask_Datastore{i,:};
        disp('Copying training data 1');
        for m=1:numel(train_Image_Datastore{i,:}.Files)
            copyfile(train_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Train1);
            copyfile(train_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Train1);
        end

    %For fold 2
    elseif i==2
        imds_train_2 = train_Image_Datastore{i,:};
        pxl_train_2 = train_Mask_Datastore{i,:};
        disp('Copying training data 2');
        for m=1:numel(train_Image_Datastore{i,:}.Files)
            copyfile(train_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Train2);
            copyfile(train_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Train2);
        end

    %For fold 3
    elseif i==3
        imds_train_3 = train_Image_Datastore{i,:};
        pxl_train_3 = train_Mask_Datastore{i,:};
        disp('Copying training data 3');
        for m=1:numel(train_Image_Datastore{i,:}.Files)
            copyfile(train_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Train3);
            copyfile(train_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Train3);
        end

    %For fold 4
    elseif i==4
        imds_train_4 = train_Image_Datastore{i,:};
        pxl_train_4 = train_Mask_Datastore{i,:};
        disp('Copying training data 4');
        for m=1:numel(train_Image_Datastore{i,:}.Files)
            copyfile(train_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Train4);
            copyfile(train_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Train4);
        end

    %For fold 5
    elseif i==5
        imds_train_5 = train_Image_Datastore{i,:};
        pxl_train_5 = train_Mask_Datastore{i,:};
        disp('Copying training data 5');
        for m=1:numel(train_Image_Datastore{i,:}.Files)
            copyfile(train_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Train5);
            copyfile(train_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Train5);
        end
    end

    validation_Image_Datastore{i,:} = subset(inputIMG, val_idx);
    validation_Mask_Datastore{i,:} = subset(outputIMG, val_idx);

    %Create an imageDatastore object to store the validation images and mask
    % for each k-fold

    %For fold 1
    if i==1
        imds_val_1 = validation_Image_Datastore{i,:};
        pxl_val_1 = validation_Mask_Datastore{i,:};
        disp('Copying val files to k1 folder validation');
        for m=1:numel(validation_Image_Datastore{i,:}.Files)
            copyfile(validation_Image_Datastore{1, 1}.Files{m, 1}, inputPath_Val1);
            copyfile(validation_Mask_Datastore{1, 1}.Files{m, 1}, outputPath_Val1);
        end

    %For fold 2
    elseif i==2
        imds_val_2 = validation_Image_Datastore{i,:};
        pxl_val_2 = validation_Mask_Datastore{i,:};
        disp('Copying val files to k2 folder');

        for m=1:numel(validation_Image_Datastore{i,:}.Files)
            copyfile(validation_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Val2);
            copyfile(validation_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Val2);
        end

    %For fold 3
    elseif i==3
        imds_val_3 = validation_Image_Datastore{i,:};
        pxl_val_3 = validation_Mask_Datastore{i,:};
        disp('Copying val files to k3 folder');

        for m=1:numel(validation_Image_Datastore{i,:}.Files)
            copyfile(validation_Image_Datastore{1, 1}.Files{m, 1}, inputPath_Val3);
            copyfile(validation_Mask_Datastore{1, 1}.Files{m, 1}, outputPath_Val3);
        end

    %For fold 4
    elseif i==4
        imds_val_4 = validation_Image_Datastore{i,:};
        pxl_val_4 = validation_Mask_Datastore{i,:};
        disp('Copying val files to k4 folder');

        for m=1:numel(validation_Image_Datastore{i,:}.Files)
            copyfile(validation_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Val4);
            copyfile(validation_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Val4);
        end
    
    %For fold 5
    elseif i==5
        imds_val_5 = validation_Image_Datastore{i,:};
        pxl_val_5 = validation_Mask_Datastore{i,:};
        disp('Copying val files to k5 folder');

        for m=1:numel(validation_Image_Datastore{i,:}.Files)
            copyfile(validation_Image_Datastore{i, 1}.Files{m, 1}, inputPath_Val5);
            copyfile(validation_Mask_Datastore{i, 1}.Files{m, 1}, outputPath_Val5);
        end
    end
end

