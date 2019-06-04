function word_histograms = word_mining(images, vocabulary, patch_dim, no_extract_words)
% This function extracts word histograms on the given images.
% Input:
% - images: images
% - vocabulary: [pxw] vocabulary (p: no. features per word, w: no. words)
% - patch_dim: visual word dimensions (sqrt(p))
% - no_extract_words: how many visual words to be extracted (per image)
%Author: Nikolas Tsagkopoulos

[~, no_words] = size(vocabulary);

% the required return
%extracted_patches = cell(no_extract_words, size(images,1));
word_histograms = zeros(no_words, size(images,1));
fprintf('Extracting histograms.. \n');

for i=1:size(images,1)
    [height, width] = size(images{i}); % height should be always the same as width
    distances = zeros(1,no_words);
    distance_per_word = zeros(1,no_extract_words);
    for p=1:no_extract_words
        pins = randi([1 width],1,2); % pins: upper left corner of visual word 
        while (pins(1) > height - 8) || (pins(2) > width - 8)
            pins = randi([1 width],1,2); %ensure its always possible to extract a visual word
        end
        extracted_patches = images{i}(pins(1):pins(1) + patch_dim - 1, pins(2):pins(2) + patch_dim - 1);
        extracted_patches = reshape(extracted_patches',1,[]);
        
        %Get the nearest centroid
        for k=1:no_words
            distances(k) = vl_alldist2(extracted_patches', vocabulary(:,k),'l1');
        end
        [~, idx] = ismember(min(distances), distances);
        distance_per_word(p) = idx;
    end
    
    a = arrayfun(@(z) sum(ismember(distance_per_word,z)),1:no_words)';
    out = [(1:no_words)' a 100*a/numel(distance_per_word)];
    word_histograms(:,i) = out/sum(out);
end
fprintf('Done! \n');

end



