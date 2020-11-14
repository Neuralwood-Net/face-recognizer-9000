# face-recognizer-9000
Project repository for the group project in NTNU course TDT4173 - Machine Learning, Fall 2020. The `face-regognizer-9000` is a Convolutional Neural Network trained for facial recognition, for use in entry access control. 

Made by Lars Ankile, Morgan Heggland and Kjartan Krange.

![Image of WoodNet](https://i.imgur.com/8PsWjr8.png)

# Setup
- Most of the code is run in notebooks using virtual machines and require no setup other than access to a virtual machine (e.g. Google Colab or Google Cloud VMs). The notebooks can be found in `notebooks`, containing both data pre-processing and model implementation and training.


## Live webcam demo setup
- Clone the repo using `git clone`
- Enter the project directory with `cd face-regognizer-9000`
- Install a virtual environment for the project:`python3 -m venv /path/to/new/virtual/environment`
- Activate virtual environment with`source /path/to/new/virtual/environment/bin/activate`
- Install depenencies needed to run the demo with `pip install -r src/demo/requirements.txt`
- Start the demo by running`python3 src/demo/demo.py` (requires a camera connected to the computer)

Note: the demo runs on CPU and thus run with a rather limited framerate. It is possible to run on gpu by editing the code where the facial recognition model is loaded in `recognizer.py` (requires GPU + CUDA installed and pyTorch configured for GPU).

# Data
The data used to train the models in this repository (beyond pre-trained models used for transfer learning) have been collected by the authors and is stored in Google Cloud Storage.

# File structure
- `models` - Storage of built, persisted models used in the project.
  - `model_1`
  - ...
- `results` - Documentation of how the models are performing on the datasets, including metrics such as precision, recall, f1-measure etc.
  - `model_1`
  - ...
- `notebooks` - Notebooks used for testing, data exploration etc.
- `src` - The core source code for the project.
  - `preprocessing` - Group of scripts that handle processing of the data.
  - `models` - Source code for building the models.
  - `demo` - Directory containing implementation of a live classification demo using a connected webcam feed.
  - `visualization` - Code for visualization for evaluation of models.
