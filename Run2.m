%%%%%% IMAGE CLASSIFICATION USING BAG OF VISUAL WORDS %%%%%%%%
%Author: Nikolas Tsagkopoulos

%LOADING TRAIN IMAGES
train_path = strcat(cd,'/training/');
tr_data = imageDatastore(train_path,'IncludeSubfolders',true,'LabelSource','foldernames');
train_images = readall(tr_data);
train_labels = cellstr(tr_data.Labels);


%LOADING TEST IMAGES
test_path = strcat(cd,'/testing/');
ts_data = imageDatastore(test_path);
test_images = readall(ts_data);

new_dim = 256; % Resizing image convenient and satisfying..

%Crop and resize images 

for i=1:size(train_images,1)
    %square it using the min dimension
    [height, width] = size(train_images{i});
    square = min([height width]);
    %crop
    crop_region = [floor((width - square)/2), floor((height - square)/2),  square, square];
    cropped_im = imcrop(train_images{i}, crop_region);
    %resize
    train_images{i,1} = im2double(imresize(cropped_im,[new_dim new_dim]));
end

for i=1:size(test_images,1)
    %square it using the min dimension
    [height, width] = size(test_images{i});
    square = min([height width]);
    %crop
    crop_region = [floor((width - square)/2), floor((height - square)/2),  square, square];
    cropped_im = imcrop(test_images{i}, crop_region);
    %resize
    test_images{i,1} = im2double(imresize(cropped_im,[new_dim new_dim]));
end


% BUILD VOCABULARY

words = 500; %vocabulary size
words_per_image = 1000; %how many visual words to extract per image
patch_dim = 8; % visual word dimension
vocabulary = bag_of_words(train_images, words, patch_dim, words_per_image);
%load('vocabulary500.mat'); %Instead of building new vocabulary

% Extract words from images
no_extract_words = 100; %how many visual words to extract per image
extracted_words_tr = word_mining(train_images,vocabulary, patch_dim, no_extract_words)';
extracted_words_ts = word_mining(test_images, vocabulary, patch_dim, no_extract_words)';

%Accuracy based on K-fold validation
k_fold = 5;
k_fold_ind = randperm(size(extracted_words_tr,1));
partition_size = size(extracted_words_tr,1)/k_fold;
accuracy = zeros(k_fold,1);

LAMBDA = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10];
for val = LAMBDA
    for k=1:k_fold
        ind = k_fold_ind((k-1)*partition_size + 1:(k-1)*partition_size+partition_size);
        ts_samples = extracted_words_tr(ind,:);
        ts_labels = train_labels(ind);
        tr_samples = extracted_words_tr(setdiff(1:size(extracted_words_tr,1),ind),:);
        tr_labels = train_labels(setdiff(1:size(extracted_words_tr,1),ind));

        predicted_categories = svm_classify(tr_samples, tr_labels, ts_samples, val);
        count = strcmp(ts_labels,predicted_categories);
        accuracy(k) = sum(count==1)/size(ts_samples,1);
    end
    printStr = sprintf('Accuracy for LAMBDA = %i on %i-fold validation: %.2f %%.', val, k_fold, sum(accuracy)*100/k_fold);
    disp(printStr);
end

% Acurracy on the tested data
LAMBDA = 0.00001;
predicted_categories = svm_classify(extracted_words_tr, train_labels, extracted_words_ts,LAMBDA);

fileID = fopen('Run2_predictions.txt','w');
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



