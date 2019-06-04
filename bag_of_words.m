function vocabulary = bag_of_words(images, words, patch_dim, words_per_image)
% Build visual word vocabulary.
% Inputs:
% - images: images
% - words: vocabulary length
% - path_dim: dimensions of visual words
% - words_per_image: No. of words extracted per image
% Author: Nikolas Tsagkopoulos

patches = cell(words_per_image, size(images,1));

printStr = sprintf('Extracting words from training samples...');
fprintf('Extracting words from training samples... \n');
for i=1:size(images,1)
    [height, width] = size(images{i}); % height should be always the same as width
    for p=1:words_per_image
        pins = randi([1 width],1,2); % pins: upper left corner of visual word 
        while (pins(1) > height - 8) || (pins(2) > width - 8)
            pins = randi([1 width],1,2); %ensure its always possible to extract a visual word
        end
        patches{p,i} = images{i}(pins(1):pins(1) + patch_dim - 1, pins(2):pins(2) + patch_dim - 1);
        patches{p,i} = reshape(patches{p,i}',1,[]);
    end
end

fprintf('Done! \n');
fprintf('Creating vocabulary.. \n');

patches = cell2mat(reshape(patches,[],1));
[vocabulary, ~] = vl_kmeans(patches', words);%, 'distance', 'l1', 'algorithm', 'elkan');
fprintf('Done! \n')'
end