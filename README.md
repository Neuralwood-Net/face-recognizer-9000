# face-recognizer-9000
Project repository for the group project in NTNU course TDT4173 - Machine Learning, Fall 2020. The face-regognizer-9000 is a convolutional neural network trained for facial recognition, for use in entry access control. 

Made by Lars Ankile, Morgan Heggland and Kjartan Krange.

# Setup
(To be revised) 
- Clone the repo using `git clone`
- `cd face-regognizer-9000`
- `python3 -m venv /path/to/new/virtual/environment`
- `source <venv>/bin/activate`
- `pip install -r requirements.txt`

# Data
[Brief description  of data + where it is stored / can be accessed if we do not store it in this repository]

# File structure
Subject to change. Proposed structure:
- `data` - The datasets used for training and validation purposes. 
  - `raw`
  - `processsed`
  - `clean`
- `models` - Storage of built, persisted models used in the project.
  - `model_1`
  - ...
- `results` - Documentation of how the models are performing on the datasets, including metrics such as precision, recall, f1-measure etc.
  - `model_1`
  - ...
- `notebooks` - Notebooks used for testing, data exploration etc.
- `source` - The core source code for the project.
  - `data` - Group of scripts that handle processing of the data.
  - (`features`) - If relevant: Set of scripts for feature extraction, or a folder for similar work such as optimzation.
  - `model` - Source code for building the models.
  - `visualization` - Code for visualization for evaluation of models.
