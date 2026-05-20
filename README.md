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
To verify that your MATLAB environment and required toolboxes are correctly set up, run the following synthetic test. No external data files are needed.

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
```

Expected output: accuracy > 0.85 (typically 0.90–0.95). This confirms that your MATLAB installation has the Statistics and Machine Learning Toolbox and that the basic SVM workflow functions correctly.

Example Data File (Optional)
This repository does not include actual field data (due to third‑party restrictions). However, you can create minimal example files in sample_data/ to test the end‑to‑end pipeline:

sample_input.xlsx – columns: 井号, AC, CAL, CN, DEN, 岩性 (at least 3 rows).

sample_norm_params.xlsx – columns: 井号, AC_AVG, AC_STD, CAL_AVG, CAL_STD, CN_AVG, CN_STD, DEN_AVG, DEN_STD.

Then run fracture_predict.m on sample_input.xlsx to see the full prediction workflow.

## 3. Repository Structure
.
├── training_scripts/               # SVM training for different lithologies
│   ├── SVM_train.m                 # 4‑log model (mudstone)
│   ├── SVM_train2.m                # 3‑log model (mudstone)
│   ├── SVM_Xsandtrain.m            # 4‑log (fine sandstone)
│   ├── SVM_Xsandtrain2.m           # 3‑log (fine sandstone)
│   ├── SVM_ZCsandtrain.m           # 4‑log (medium‑coarse sandstone)
│   ├── SVM_ZCsandtrain2.m          # 3‑log (medium‑coarse sandstone)
│   ├── SVM_Fsandtrain.m            # 4‑log (siltstone)
│   ├── SVM_Fsandtrain2.m           # 3‑log (siltstone)
│   ├── SVM_SLsandtrain.m           # 4‑log (glutenite)
│   ├── SVM_SLsandtrain2.m          # 3‑log (glutenite)
│   ├── SVM_mud3train.m             # 4‑log (mudstone, extended)
│   ├── SVM_mud4train2.m            # 3‑log (mudstone, extended)
│   ├── SVM_baiyunyan3train.m       # 4‑log (dolomite)
│   ├── SVM_baiyunyan4train2.m      # 3‑log (dolomite)
│   ├── SVM_Bmudtrain.m             # 4‑log (dolomitic mudstone)
│   ├── SVM_Bmudtrain2.m            # 3‑log (dolomitic mudstone)
│   ├── SVM_NZbaiyunyantrain.m      # 4‑log (argillaceous dolomite)
│   ├── SVM_NZbaiyunyantrain2.m     # 3‑log (argillaceous dolomite)
│   └── ... (other lithology variants)
├── prediction_scripts/
│   ├── fracture_predict.m          # Predict using 4‑log models (AC,CAL,CN,DEN)
│   ├── fracture_predict2.m         # Predict using 3‑log models (AC,CAL,SP)
│   ├── fracture_predict3.m         # Advanced: 8 lithology classes, 4‑log
│   └── fracture_predict4.m         # Advanced: 8 lithology classes, 3‑log (with fixed normalisation)
├── utils/
│   ├── Data_Std.m                  # Z‑score normalisation example
│   └── Bayes_Classification.m      (alternative Bayesian classifier – not used in final paper)
├── sample_data/                    (not included, see Section 2)
│   ├── sample_input.xlsx
│   ├── sample_norm_params.xlsx
│   └── ...
└── README.md

## 4. Data Preparation

### 4.1 Input files for training
Each training script requires two Excel files:

Sample file (e.g., 泥岩裂缝样本.xlsx)

First row: headers. At least the following columns must exist:
井号 (well ID), AC, CAL, CN, DEN (or SP for 3‑log models), and LABEL (1 = fracture, 0 = non‑fracture).

Each subsequent row corresponds to a depth point with known fracture label (from core description).

Normalisation parameters file (e.g., 归一化参数.xlsx)

First row: headers.

Columns: 井号, AC_AVG, AC_STD, CAL_AVG, CAL_STD, CN_AVG, CN_STD, DEN_AVG, DEN_STD (or SP_AVG, SP_STD for 3‑log).

Each row provides the mean and standard deviation of each log for that specific well (calculated over the target interval). These are used for well‑by‑well Z‑score normalisation.

### 4.2 Input files for prediction
The prediction scripts (fracture_predict.m etc.) ask the user to select an Excel file containing the logs to predict. The file must include:

A header row with column names.

Columns: 井号 (well ID – can be repeated), AC, CAL, CN, DEN (or SP), and 岩性 (lithology string).

Supported lithology strings (case‑insensitive partial match):
砂砾岩, 中-粗砂岩, 细砂岩, 粉砂岩, 泥岩, 白云质泥岩, 白云岩, 泥质白云岩.

The prediction script automatically detects which log curves are present and uses the appropriate model (4‑log or 3‑log). It also performs automatic normalisation using the mean and standard deviation of the input logs (or uses a fixed pre‑defined normalisation when auto_set_std_param = false).

## 5. Training a New SVM Model

### 5.1 Four‑log model (AC, CAL, DEN, CN)

Example for mudstone:

Prepare 泥岩裂缝样本.xlsx and 归一化参数.xlsx as described.

Open SVM_train.m and set the correct file names and target_index = ["AC", "CAL", "CN", "DEN"].

Run the script. It will:

Read and normalise the data per well.

Split into training (70%) and test (30%) while preserving class proportions.

Perform 5‑fold cross‑validation with Bayesian optimisation to find optimal sigma (kernel scale) and box (box constraint).

Train the final SVM with RBF kernel and compute posterior probabilities.

Save the model as SVM_model_mud.mat.

Display confusion matrix, accuracy, recall, precision, F1 score on the test set.

Generate a plot of predicted vs. true labels.

### 5.2 Three‑log model (AC, CAL, SP)

Use SVM_train2.m and change target_index = ["AC", "CAL", "SP"]. The normalisation parameters file must contain SP_AVG and SP_STD.

### 5.3 For other lithologies

Simply duplicate the script, change the Excel file names, and adjust Model_save_path and target_index accordingly. The repository already contains pre‑trained models for eight lithologies (see fracture_predict3.m for the full list).

## 6. Predicting on New Wells

### 6.1 Basic prediction (one model type)

Four‑log prediction: run fracture_predict.m.

Three‑log prediction: run fracture_predict2.m.

The script will open a file dialog – select your well log Excel file. It then:

Automatically detects which log curves are available.

Normalises the logs (either per‑file mean/std or using fixed parameters if auto_set_std_param = false).

Classifies each depth point according to its lithology using the appropriate pre‑trained SVM model (models must be in the same folder as the script).

Outputs a new Excel file with three additional columns:

SVM预测分类 (1 = fracture, 0 = non‑fracture)

SVM分类概率 (maximum class probability)

PNN预测分类 (reserved – if you also have PNN models)

The output file is named <original>_predict.xlsx.

### 6.2 Advanced prediction (eight lithology classes)

For the full set of eight lithologies used in the paper, use:

fracture_predict3.m (4‑log models)

fracture_predict4.m (3‑log models with optional fixed normalisation parameters)

These scripts load all eight SVM models and route each depth point to the correct classifier based on the 岩性 column.

## 7. Reproducing the Paper’s Results

The trained models and anonymised test data are not included in this repository for confidentiality reasons. However, the scripts are fully functional and can be run on your own datasets. To reproduce the performance metrics (Table 4 in the paper):

Prepare training files for each lithology with fracture labels derived from core description.

Run the corresponding training script for that lithology (e.g., SVM_ZCsandtrain.m for medium‑coarse sandstone).

The script outputs test set accuracy, recall, precision, and F1 score, along with a confusion matrix.

All figures in the paper (crossplots, single‑well logs, confusion charts) can be generated by modifying the training scripts to save the plots. Example plotting code is already included (figure, plot, confusionchart).

## 8. Example Workflow

Here is a typical sequence to train a model and predict on a new well:
% ---------- TRAINING ----------
% 1. Prepare Excel files: "my_fine_sand_samples.xlsx" and "my_norm_params.xlsx"
% 2. Edit SVM_Xsandtrain.m:
%    - train_raw = xlsread("my_fine_sand_samples.xlsx");
%    - para_raw = xlsread("my_norm_params.xlsx");
%    - Model_save_path = "SVM_model_Xsand.mat";
% 3. Run SVM_Xsandtrain.m
%    -> model saved, test set metrics printed.

% ---------- PREDICTION ----------
% 1. Prepare well log Excel: "well_A_logs.xlsx" with columns: 井号, AC, CAL, CN, DEN, 岩性
% 2. Run fracture_predict.m, select "well_A_logs.xlsx"
% 3. Output: "well_A_logs_predict.xlsx" with SVM predictions.

## 9. Customising Normalisation

The following code shows how to manually set fixed normalisation parameters (instead of automatic calculation). Save it as custom_norm.m:
%% ==================== Example: Custom Normalisation Parameters ====================
% For use in fracture_predict.m / fracture_predict2.m etc.
% Replace the corresponding section in the original script with the code block below.

auto_set_std_param = false;   % Disable automatic normalisation

% ----- 4‑parameter model (AC, CAL, CN, DEN) -----
% Order: AC_AVG, AC_STD, CAL_AVG, CAL_STD, CN_AVG, CN_STD, DEN_AVG, DEN_STD
std_param_4log = [335.9975, 15.68873, 25.47676, 1.819065, 22.52, 6.678, 2.438, 0.208];

% ----- 3‑parameter model (AC, CAL, SP) -----
% Order: AC_AVG, AC_STD, CAL_AVG, CAL_STD, SP_AVG, SP_STD
std_param_3log = [335.9975, 15.68873, 25.47676, 1.819065, 80.10648, 8.057371];

% Assign the appropriate vector to std_param according to the script used
std_param = std_param_4log;   % or std_param_3log

## 10.Citation

Xu, S., Wang, R., Fan, Z., Yao, R., Ren, S., Jiang, Y., Dong, Y., Xie, C. (2026). Micro-Fracture Identification and Prediction with Improved SVM Model: A Case Study of Complex Carbonate and Clastic Rock Reservoirs in the Niuxintuo Area, Liaohe Depression, Bohai Bay Basin. Computers & Geosciences, 214, 106183. https://doi.org/10.1016/j.cageo.2026.106183

## 11. Acknowledgements

The authors gratefully acknowledge the financial support from the 
National Natural Science Foundation of China (Grant No. 42202121).

We thank the Research Institute of Petroleum Exploration and Development, Liaohe Oilfield Company, for providing core samples and well-log data.

We appreciate the support provided by the Key Laboratory of Tectonics and Petroleum Resources at the China University of Geosciences (Wuhan).

## 12. License

This code is distributed under the MIT License. See the `LICENSE` file for details.
