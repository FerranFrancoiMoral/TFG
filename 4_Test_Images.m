clc; clear all; close all; 

%We create a file datastore with the .mat files containing the images and
%the masks
fds = fileDatastore(fullfile("D:\TFG\our_simulated_data_testData\20230426_simulatedData"),"ReadFcn",@load,"FileExtensions",".mat");

num_files = numel(fds.Files);
inputPath_Test = 'D:\TFG\our_simulated_data_testData\input_test\';
outputPath_Test = 'D:\TFG\our_simulated_data_testData\output_test\';

%% Load the images and masks from the .mat files (test folder)


for idx = 1:num_files
   
   %Store the image in the inputPath_Test folder
   file = load(fds.Files{idx,1});
   in = imresize(file.simData.BmodeInfo.BmodeImg,[256 256]);
   name=fds.Files{idx,1};
   name=name(end-9:end-4);
   imgName=append(inputPath_Test,name,'.png');
   imwrite(in , imgName);
   imageFilenames{idx,1}=imgName;

   %Create the GT mask with all the 3 masks combined
   Al = imresize(file.simData.ALinesInfo.ALMask,[256 256]);
   Bl = imresize(file.simData.BLinesInfo.BLMask,[256 256]);
   Pl = imresize(file.simData.PleuralLineInfo.PLMask,[256 256]);
   out = Al|Bl|Pl; 

   % Store the mask in the outputPath_Test folder
   maskName=append(outputPath_Test,name,'.png');
   imwrite(out , maskName);
   maskFilenames{idx,1}=maskName;
end
