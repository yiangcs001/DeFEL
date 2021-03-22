# DeFEL
---
DeFEL is a tool to predict m6A-site containing sequences.

## Requirements
---

1. python ≥ 3.6.11
2. tensorflow-gpu = 2.2.0
3. xlrd = 1.2.0
4. pyyaml = 5.4.1
5. scikit-learn = 0.23.2
6. numpy = 1.19.1
7. pandas = 1.1.1

## Installation

---

Run the source code from python directly.

## Usage

---

1. Run `python create_data_4_deep_model.py` to split data into several subsets, and each subset will be used to train a deep one-hot model and deep chemical model.
2. Run `python train_deep_model.py` to train several deep one-hot models or deep chemical models. The trained deep models are stored in directory `codes/models/DM`. If you want to used the trained deep models, you can skip this step.
3. Run `python train_DeFEL.py` to train several RF models and LR models. The trained RF models and LR models are stored in directory `codes/models/RF` and `codes/models/LR`, respectively. If you want to used the trained RF models and LR models, you can skip this step.
4. Run `python test_DeFEL.py` to predict m6A sites of test.xlsx. 

**NOTE: The default configuration file of `create_data_4_deep_model.py` is `deep_model_config.yaml`, and the default configuration file of `train_DeFEL.py` and `test_DeFEL.py` is `defel_config.yaml`. You can set the parameters in the default configuration file, or specify your configuration file while running `python train_deep_model.py/train_DeFEL.py/test_DeFEL.py` with `-f <your_file.yaml>`.**

## Copyright and License Information

---

Copyright (C) 2021 Northwestern Polytechnical University, Xi’an, China.
