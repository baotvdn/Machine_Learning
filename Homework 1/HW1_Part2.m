%% Homework 1, Part 2
% Bao Dang
clc; clear all; close all;

% Import the image files and create a 154x1600 data matrix 
D = dir ('yalefaces');
F = struct2dataset(D);
G = F.name;
H = G(4:end,:);
Matrix = [];
for i = 1:length(H)
    % Read in the image as a 2D array
    A = imread(strcat('yalefaces\',H{i}));
    % Subsample the image to become a 40x40 pixel image
    B = imresize(A, [40 40]);
    % Flatten the image to a 1D array (1x1600)
    C = reshape(B, 1, []);
    % Concatenate this as a row of your data matrix
    Matrix = [Matrix; C];
end

% Standardizes the data
Avg = mean(Matrix);
Sd = std(double(Matrix));
Standard = (double(Matrix) - repmat(Avg,size(Matrix,1),1)) ./ repmat(Sd,size(Matrix,1),1);

% Reduces the data to 2D using PCA
covariance = cov(Standard);
[W, lambda] = eig(covariance);
Z = Standard*W(:,end-1:end); %Since it is 2D, we need 2 eigen vectors that have highest eigen values

%Graphs the data for visualization
scatter(Z(:,2),Z(:,1));
title('PCA');

