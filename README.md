# proj-incremental-learning
There are two main files in this project - cls-dl-sm-controller-script.py (controller script) and classification-training.py (training script). First one is the script which invokes the second one after doing some preprocessing and later some cleaning up.
The first script invokes an instance and provides various locations of S3 locations. This script first prepares the dataset for training and saves to a bucket.
The second script downloads the data from the given bucket. It also downloads the pretrained model (has a .pth extension) locally uses the pretrained model to load the various weights before attempting to train.
Some of the parameters like number of epochs and learning rate are passed from controller to training script.
The training script trains the model and copies the model to a predefined location known as model directory. The contents of this directory are uploaded to a S3 location ("output_path") specified by the controller script.
After the training is finished, the controller script then downloads the artefacts from the S3 path and uploads this to the S3 model location (known as MODEL_BUCKET in the controller script).
Controller script then performs some houskeeping.
Things to Note: I would like you to remember some points regarding the model output. The training scripts outputs two model files, one ends with .pth and other ends with .pkl.

The .pth file is used for incremental learning and .pkl file will be used for inferencing. This .pkl file can ne served through a lambda function and a project is available for this as well. You can refer and use this project to productionize a deep learning model trained with fastai framework.
