%% Homework 1, Part 3
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

% Perform PCA
covariance = cov(Standard);
[W, lambda] = eig(covariance);

% Determines the number of principle components necessary to encode at least 95% of the information, k
lambda_element = diag(lambda);
i = length(lambda_element);
k = 0;
threshold = 0;
sumD = sum(lambda_element);
sumk = 0;
while threshold < 0.9500
    sumk = sumk + lambda_element(i);
    i = i - 1;
    k = k + 1;
    threshold = sumk/sumD;
end

% Visualizes the most important principle component as a 40x40 image
Z1 = Standard*W(:,end);
X1 = Z1*W(:,end)';
X1 = X1.*Sd + Avg;              %Unstandardize the matrix
X1 = reshape(X1(1,:),40,40);
figure('Name','Primary Principle Component');
imshow(uint8(X1));

% Reconstructs the first person using the primary principle component and then using the k most
% significant eigen-vectors
Z2 = Standard*W(:,end);
X2 = Z2*W(:,end)';
X2 = X2.*Sd + Avg;              %Unstandardize the matrix
X2 = reshape(X2(1,:),40,40);
figure('Name','Single principle component');
imshow(uint8(X2));

% k-most significant eigen-vectors
Z = Standard*W(:,i+1:end);      %Take 37 eigenvectors from 1564 to 1600
X = Z*W(:,i+1:end)';
X = X.*Sd + Avg;                %Unstandardize the matrix
X = reshape(X(1,:),40,40);
figure('Name','k principle component (k =37)');
imshow(uint8(X));
