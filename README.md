# SVM-based Fracture Prediction from Conventional Well Logs (MATLAB)

This repository contains the MATLAB source code for the paper:  
**"Micro-Fracture Identification and Prediction with Improved SVM Model: A Case Study of Complex Carbonate and Clastic Rock Reservoirs in the Niuxintuo Area, Liaohe Depression, Bohai Bay Basin"**  
by Shikun Xu, Ren Wang, Zuochun Fan, Rui Yao, Shuangpo Ren, Yue Jiang, Yang Dong, Congjiao Xie (2026, *Computers & Geosciences*).

The code implements a complete workflow for predicting natural fractures using conventional well logs (AC, DEN, CAL, CNL, SP) and a Support Vector Machine (SVM) classifier. Two sets of models are provided:
- **Four‑log model** (AC, DEN, CNL, CAL) – for wells with full logging suites.
- **Three‑log model** (AC, CAL, SP) – for wells missing DEN and CNL curves (common in older wells).

The workflow includes well‑by‑well Z‑score normalisation, Bayesian optimisation of SVM hyperparameters (box constraint and kernel scale), 5‑fold cross‑validation, SMOTE‑like balancing (via data preparation), and outputs fracture probability logs with >85% accuracy.

## 1. System Requirements

- MATLAB R2019b or later (tested on R2021b)
- Statistics and Machine Learning Toolbox (for `fitcsvm`, `bayesopt`, `cvpartition`)
- Parallel Computing Toolbox (optional, speeds up Bayesian optimisation)
- MATLAB’s `xlswrite` / `xlsread` functions (requires Excel support on your OS)

No additional third‑party toolboxes are required.

## 2. Quick Test

To verify that your MATLAB environment and required toolboxes are correctly set up, run the following synthetic test. **No external data files are needed.**

Save the script below as `quick_test.m` and run it:

```matlab
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

Expected output: accuracy > 0.85 (typically 0.90–0.95). This confirms that your MATLAB installation has the Statistics and Machine Learning Toolbox and that the basic SVM workflow functions correctly.
