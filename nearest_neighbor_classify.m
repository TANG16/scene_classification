function predicted_categories = nearest_neighbor_classify(tr_arr_feats,train_labels,ts_arr_feats,k)
% The structure of this algorithm belongs to James Hays for CS 143, Brown
% University.
% Inputs:
% - tr_arr_feats: [m x p] training samples (m: samples, p: features)
% - train_labels: labels
% - ts_arr_feats: [m x p] testing samples
% - k: No. neighbors
%Author: Nikolas Tsagkopoulos

testing_samples = size(ts_arr_feats, 1);

%Measure the distance of every ts_feat from every tr_feat
distance = vl_alldist2(tr_arr_feats', ts_arr_feats');

%Count the classes
unique_labels = unique(train_labels);
labels_size = size(unique_labels, 1);

%Sort the distances
[~, indices] = sort(distance, 1);

predicted_categories = cell(testing_samples, 1);
for i = 1 : testing_samples
    occurency_idx = 0; %indexes of the highest occured labels
    occurency_counter = 0; % counter highest occured label
    for j = 1 : labels_size
        nearest_labels = train_labels(indices(1 : k, i));
        compare = sum(strcmp(unique_labels(j), nearest_labels)); %how many labels occures in the knn
        if (compare > occurency_counter)
            occurency_idx = j;
            occurency_counter = compare;
        end
    end
    predicted_categories{i, 1} = unique_labels{occurency_idx}; %defining the label
end

end