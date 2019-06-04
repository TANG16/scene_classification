%%%%%% IMAGE CLASSIFICATION THROUGH K-NEAREST NEIGHBORS %%%%%%%%
%Authors: Nikolas Tsagkopoulos

close all; clear all; clc;
%LOAD TRAINING IMAGES
train_path = strcat(cd,'/training/');
tr_data = imageDatastore(train_path,'IncludeSubfolders',true,'LabelSource','foldernames');
train_images = readall(tr_data);
train_labels = cellstr(tr_data.Labels);

%LOAD TESTING IMAGES
test_path = strcat(cd,'/testing/');
ts_data = imageDatastore(test_path);
test_images = readall(ts_data);
%test_labels = ts_data.Labels;

tiny_image = 16;

%OBTAINING TRAINING TINY FEATURES
tr_arr_feats = zeros(size(train_images,1),tiny_image^2);
for i=1:size(train_images,1)
    %square it using the minimum dimension
    [height, width] = size(train_images{i});
    square = min([height width]);
    %crop
    crop_region = [floor((width - square)/2), floor((height - square)/2),  square, square];
    cropped_im = imcrop(train_images{i}, crop_region);
    %get tiny image
    resized_im = im2double(imresize(cropped_im,[tiny_image tiny_image]));
    tr_arr_feats(i,:) = reshape(resized_im',1,[]);
    tr_arr_feats(i,:) = (tr_arr_feats(i,:)-min(tr_arr_feats(i,:)))/(max(tr_arr_feats(i,:))-min(tr_arr_feats(i,:))); %normalizing (unit length)
    tr_arr_feats(i,:) = bsxfun(@minus, tr_arr_feats(i,:), mean(tr_arr_feats(i,:))); % 0 centered
end

%Proof that every feature corresponds to true image
%figure; imagesc(reshape(tr_arr_feats(1,:),[tiny_image, tiny_image])');
%figure; imagesc(train_images{1});

%OBTAINING TESTING TINY FEATURES
ts_arr_feats = zeros(size(test_images,1),tiny_image^2);
for i=1:size(test_images,1)
    %square it using the min dimension
    [height, width] = size(test_images{i});
    square = min([height width]);
    %crop
    crop_region = [floor((width - square)/2), floor((height - square)/2),  square, square];
    cropped_im = imcrop(test_images{i}, crop_region);
    %get tiny image
    resized_im = im2double(imresize(cropped_im,[tiny_image tiny_image]));
    ts_arr_feats(i,:) = reshape(resized_im',1,[]);
    ts_arr_feats(i,:) = (ts_arr_feats(i,:)-min(ts_arr_feats(i,:)))/(max(ts_arr_feats(i,:))-min(ts_arr_feats(i,:))); %normalizing (unit length)
    ts_arr_feats(i,:) = bsxfun(@minus, ts_arr_feats(i,:), mean(ts_arr_feats(i,:))); % 0 centered
end

%Accuracy based on K-fold validation
neighbors = 1;
k_fold = 5;
k_fold_ind = randperm(size(tr_arr_feats,1));
partition_size = size(tr_arr_feats,1)/k_fold;
accuracy = zeros(k_fold,1);

for k=1:k_fold
    ind = k_fold_ind((k-1)*partition_size + 1:(k-1)*partition_size+partition_size);
    ts_samples = tr_arr_feats(ind,:);
    ts_labels = train_labels(ind);
    tr_samples = tr_arr_feats(setdiff(1:size(tr_arr_feats,1),ind),:);
    tr_labels = train_labels(setdiff(1:size(tr_arr_feats,1),ind));

    predicted_categories = nearest_neighbor_classify(tr_samples, tr_labels, ts_samples, neighbors);
    count = strcmp(ts_labels,predicted_categories);
    accuracy(k) = sum(count==1)/size(ts_samples,1);
end
    printStr = sprintf('Accuracy for %i neighbors on %i-fold validation: %.2f %%.', neighbors,k_fold, sum(accuracy)*100/k_fold);
    disp(printStr);
    

%Accuracy on the tested data
 predicted_categories = nearest_neighbor_classify(tr_arr_feats,train_labels,ts_arr_feats,neighbors);
 fileID = fopen('Run1_predictions.txt','w');
 for i=1:size(test_images,1)
     fprintf(fileID,'%i.jpeg %s\r\n',(i-1),predicted_categories{i});
 end
 fclose(fileID);


%Demonstration
figure(1);
for i=1:25
    subplot(5,5,i)
    imshow(uint8(test_images{i}));
    title(predicted_categories{i});
end






