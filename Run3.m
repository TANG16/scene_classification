%%%%%% IMAGE CLASSIFICATION USING FEATURE EXTRACTION TRAINING %%%%%%
%Author: Nikolas Tsagkopoulos

% Add MatConvNet to path.
addpath(strcat(userpath,'/matconvnet-1.0-beta25/matlab/'))
run 'vl_setupnn.m';

%LOADING TRAINING IMAGES
tr_data =imageDatastore(strcat(cd,'/training/'),'IncludeSubfolders',true,'LabelSource','foldernames');
train_images = readall(tr_data);
train_labels = string(tr_data.Labels);

%LOADING TESTING IMAGES
ts_data =imageDatastore(strcat(cd,'/testing/'));
test_images = readall(ts_data);

%More pre-trained nets can be downloaded from here:
% http://www.vlfeat.org/matconvnet/pretrained/

% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-vgg-m-1024.mat');
net = vl_simplenn_tidy(net);
feat_size = 1024; % Check the size of the features before the softmax layer!
last_layer = 20; %From which layer you will extract the features (depends on the net).

%Feeding the net with the TRAINING data
%If you change the model you have to know what your doing from now on.
extracted_training_feat = zeros(size(train_images,1),feat_size);
for i=1:size(train_images,1)
    % Obtain and preprocess a training image.
    im = cat(3, train_images{i}, train_images{i}, train_images{i}); %Fixing channels
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % Feed image to CNN.
    res = vl_simplenn(net, im_) ;

    %FEATURE EXTRACTION FROM LAST FC LAYER
    featureVector = res(last_layer).x; %Depends on the pretrained model. 
    extracted_training_feat(i,:) = featureVector (:)';
end

%Feeding the net with the TESTING data
extracted_testing_feat = zeros(size(test_images,1),feat_size);
for i=1:size(test_images,1)
    % Obtain and preprocess a testing image.
    im = cat(3, test_images{i}, test_images{i}, test_images{i});
    im_ = single(test_images{i}) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % Feed image to CNN.
    res = vl_simplenn(net, im_) ;

    %FEATURE EXTRACTION FROM THE LAST LAYER
    featureVector = res(last_layer).x;
    extracted_testing_feat(i,:) = featureVector (:)';
end

%Accuracy based on K-fold validation
k_fold = 5;
k_fold_ind = randperm(size(extracted_training_feat,1));
partition_size = size(extracted_training_feat,1)/k_fold;
accuracy = zeros(k_fold,1);

LAMBDA = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10];
for val = LAMBDA
    for k=1:k_fold
        ind = k_fold_ind((k-1)*partition_size + 1:(k-1)*partition_size+partition_size);
        ts_samples = extracted_training_feat(ind,:);
        ts_labels = train_labels(ind);
        tr_samples = extracted_training_feat(setdiff(1:size(extracted_training_feat,1),ind),:);
        tr_labels = train_labels(setdiff(1:size(extracted_training_feat,1),ind));

        predicted_categories = svm_classify(tr_samples, tr_labels, ts_samples, val);
        count = strcmp(ts_labels,predicted_categories);
        accuracy(k) = sum(count==1)/size(ts_samples,1);
    end
    printStr = sprintf('Accuracy for LAMBDA = %i on %i-fold validation: %.2f %%.', val, k_fold, sum(accuracy)*100/k_fold);
    disp(printStr);
end


% Acurracy on the tested data
LAMBDA = 0.0001;
predicted_categories = svm_classify(extracted_training_feat, train_labels, extracted_testing_feat,LAMBDA);

fileID = fopen('Run3_predictions.txt','w');
for i=1:size(test_images,1)
    fprintf(fileID,'%i.jpeg %s\r\n',(i-1),predicted_categories{i});
end
fclose(fileID);

figure(1);
for i=1:25
    subplot(5,5,i)
    imshow(test_images{i});
    title(predicted_categories{i});
end
