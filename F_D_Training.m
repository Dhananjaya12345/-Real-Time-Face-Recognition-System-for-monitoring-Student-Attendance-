clc
clear all
close all
warning off

% Load the pre-trained AlexNet
g = alexnet;
layers = g.Layers;

% Modify the fully connected and classification layers
layers(23) = fullyConnectedLayer(2);
layers(25) = classificationLayer;

% Define the path to your image datastore
dataPath = '/MATLAB Drive/5FTC1213_Lab1-1/FAce_detection/Datastorage';

% Create an imageDatastore with resized and preprocessed images
inputSize = [227, 227, 3]; % Resize images to match the expected input size
allImages = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(filename)readAndPreprocessImage(filename, inputSize));

% Define training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'Verbose', true);

% Train the network
myNet1 = trainNetwork(allImages, layers, opts);

% Save the trained network
save myNet1;

function img = readAndPreprocessImage(filename, inputSize)
    % Read an image from a file, resize it to the desired dimensions,
    % and preprocess it for the network
    img = imread(filename);
    img = imresize(img, inputSize(1:2));
    if size(img, 3) == 1
        img = cat(3, img, img, img); % Convert grayscale to RGB
    end
    img = im2single(img);
end