#This repository contains the MATLAB source code for the paper:  
**"Micro-Fracture Identification and Prediction with Improved SVM Model: A Case Study of Complex Carbonate and Clastic Rock Reservoirs in the Niuxintuo Area, Liaohe Depression, Bohai Bay Basin"** by Shikun Xu, Ren Wang, Zuochun Fan, Rui Yao, Shuangpo Ren, Yue Jiang, Yang Dong, Congjiao Xie (2026, *Computers & Geosciences*).

The code implements a complete workflow for predicting natural fractures using conventional well logs (AC, DEN, CAL, CNL, SP) and a Support Vector Machine (SVM) classifier. Two sets of models are provided:
- **Four‑log model** (AC, DEN, CNL, CAL) – for wells with full logging suites.
- **Three‑log model** (AC, CAL, SP) – for wells missing DEN and CNL curves (common in older wells).

The workflow includes well‑by‑well Z‑score normalisation, Bayesian optimisation of SVM hyperparameters (box constraint and kernel scale), 5‑fold cross‑validation, SMOTE‑like balancing (via data preparation), and outputs fracture probability logs with >85% accuracy.

##1. System Requirements
- MATLAB R2019b or later (tested on R2021b)
- Statistics and Machine Learning Toolbox (for `fitcsvm`, `bayesopt`, `cvpartition`)
- Parallel Computing Toolbox (optional, speeds up Bayesian optimisation)
- MATLAB’s `xlswrite` / `xlsread` functions (requires Excel support on your OS)

No additional third‑party toolboxes are required.
