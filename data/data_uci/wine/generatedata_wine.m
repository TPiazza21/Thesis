rand('seed', 100000);
randn('seed', 100000);
data = textread('winequality-red.csv');
[m, n] = size(data)
p = randperm(m);
data = data(p, :);
feature = data(:, [1:end-2, end]);
feature = bsxfun(@minus, feature, mean(feature));
if normalize_coordinates
    feature = bsxfun(@rdivide, feature, std(feature));
end
y = data(:, end-1);
y = y - mean(y);
data = [feature, y];
k_cv_out = 10; % k fold cross validation
cvo = cvpartition(m, 'k', k_cv_out);
save wine.mat data cvo