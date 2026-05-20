% quick_test.m – Synthetic test for SVM fracture prediction workflow
clear; close all; clc;

fprintf('=== Quick Test: Synthetic Fracture Data ===\n');

% Generate synthetic logs for 200 depth points
rng(42); % for reproducibility
n = 200;
AC  = 300 + 20*randn(n,1);   % μs/m
CAL = 24  + 2*randn(n,1);    % cm
CN  = 10  + 5*randn(n,1);    % %
DEN = 2.5 + 0.1*randn(n,1);  % g/cm³

% Create synthetic labels: fractures (1) when AC>320 & CAL>25 & CN>12 & DEN<2.45
label = (AC > 320 & CAL > 25 & CN > 12 & DEN < 2.45);
label = double(label);

% Split into training (70%) and test (30%)
cv = cvpartition(label, 'HoldOut', 0.3);
x = [AC, CAL, CN, DEN];
x_train = x(cv.training,:); y_train = label(cv.training);
x_test  = x(cv.test,:);     y_test  = label(cv.test);

% Train SVM with RBF kernel (no Bayesian optimisation for quick test)
SVMModel = fitcsvm(x_train, y_train, 'KernelFunction', 'rbf', ...
                   'Standardize', true, 'BoxConstraint', 1, 'KernelScale', 'auto');

% Predict and evaluate
y_pred = predict(SVMModel, x_test);
acc = sum(y_pred == y_test) / length(y_test);

fprintf('Test accuracy: %.3f\n', acc);
if acc > 0.85
    fprintf('=== Quick Test PASSED ===\n');
else
    fprintf('=== Quick Test FAILED (low accuracy) ===\n');
end
