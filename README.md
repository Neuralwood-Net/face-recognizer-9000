# WoodNet
Project repository for the group project in NTNU course TDT4173 - Machine Learning, Fall 2020. WoodNet¹ is a Convolutional Neural Network trained for facial recognition, for use in entry access control. 

Made by Lars Ankile, Morgan Heggland and Kjartan Krange.

![Image of WoodNet](https://i.imgur.com/8PsWjr8.png)

¹The network is named after Fleetwood Mac, as one of the authors recently discovered their enchanting tunes².  
²The other authors have been acquainted with Fleetwood Mac for a while.

# Setup
The code used to process the data and train the models is run in notebooks using virtual machines and require no setup other than access to a virtual machine (e.g. Google Colab or Google Cloud VMs). The notebooks can be found in `notebooks`, containing both data pre-processing (`data-extraction.ipynb`) and model implementation and training (`Train_plot_and_save_224px_color.ipynb`).


## Live webcam demo setup
To run the accompanying live inference demo, follow the steps below:
- Clone the repo using `git clone`
- Enter the project directory with `cd face-regognizer-9000`
- Install a virtual environment for the project:`python3 -m venv /path/to/new/virtual/environment`
- Activate virtual environment with`source /path/to/new/virtual/environment/bin/activate`
- Install depenencies needed to run the demo with `pip install -r src/demo/requirements.txt`
- Start the demo by running`python3 src/demo/demo.py` (requires a camera connected to the computer)

Note: the demo runs on CPU and thus run with a rather limited framerate. It is possible to run on GPU by editing the code where the facial recognition model is loaded in `recognizer.py`. This requires a GPU + CUDA installed and pyTorch configured for GPU, and is not tested.

PS: the demo attempts to classify faces present in the webcam according to one of the clases `Lars`, `Morgan`, `Kjartan`and `Other`. In a perfect world, this would require one of the authors to be present when you run the demo in order for it to predict anything other than `Other`.

# Data
The data used to train the models in this repository (beyond pre-trained models used for transfer learning) have been collected by the authors and is stored in Google Cloud Storage: 
- [Raw images](https://storage.googleapis.com/tdt4173-datasets/faces/images/raw_images.zip) (5.2 GB)
- [Dataset cropped using face detection](https://storage.googleapis.com/tdt4173-datasets/faces/balanced_sampled_cropped_224px_color_70_15_15_split.tar.gz) (3.1 GB)
- [Dataset cropped around center of image](https://storage.googleapis.com/tdt4173-datasets/faces/balanced_sampled_cropped_224px_color_70_15_15_split.tar.gz) (3.3 GB)

# File structure
- `models` - Storage of built, persisted models used in the project.
  - `model_1`
  - ...
- `notebooks` - Notebooks used for testing, data exploration, data processing and training.
- `src` - The core source code for the project.
  - `models` - Source code defining the models.
  - `demo` - Directory containing implementation of a live classification demo using a connected webcam feed.
  - `visualization` - Code for visualization for evaluation of models.
