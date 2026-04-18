# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational Physics 2026 — research notebook collection for astrophysical machine learning.

## Python environment
- This repo must be run with the conda comp_phys environment located in /home/zingales/anaconda3/envs/comp_phys/bin/python

## Structure

- `Astrophysical-Objects-Recognition/stellar-classification-and-supervised-learning.ipynb` — Stellar object classification notebook. Loads SDSS star classification CSV data, performs preprocessing (outlier removal, SMOTE oversampling, standardization), trains multiple ML models (LinearSVC, DecisionTree, RandomForest, CatBoost, LightGBM), a PyTorch neural network, and a VotingClassifier ensemble. Includes SHAP interpretability analysis and permutation importance.
- `Moon-Recognition/` — Empty directory, planned for future moon recognition work.

## Working with Notebooks

- This repo contains no build system, tests, or configuration files.
- Notebooks are developed standalone and can be run with `jupyter notebook` or `jupyter lab`.
- Dependencies: numpy, pandas, scikit-learn, imbalanced-learn, torch, catboost, lightgbm, matplotlib, seaborn, shap.
