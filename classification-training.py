import os
import re
import boto3
import sagemaker
import zipfile
import shutil
import time
import logging
import glob
#from fastai.text.all import *
import argparse
import math
import os
from shutil import copy
import time
from fastai.vision.all import *

NUM_EPOCHS = 1
LEARNING_RATE = 2e-3
TRAINING_PATH = ""
TOKEN_PATH = ""

def copy_generated_model(SRC_DIR, SRC_FILE, DEST_DIR):
    src = SRC_DIR + "/" + SRC_FILE
    dest = DEST_DIR + "/"
    try:
        shutil.copy(src, dest)
    except Exception as e:
        print("file copy failed:", str(e))
        
def get_init_model(SRC_MODEL_DIR, TGT_MODEL_DIR):
    ret_val = ""
    try:
        if (len(os.listdir(SRC_MODEL_DIR)) == 1):
            ret_val = os.listdir(SRC_MODEL_DIR)[0] # model file name
            if not os.path.exists(TGT_MODEL_DIR):
                os.makedirs(TGT_MODEL_DIR)
            shutil.copy(SRC_MODEL_DIR + "/" + ret_val, TGT_MODEL_DIR + "/" + ret_val )
    except Exception as exp:
        print('exception in get_init_model: ', str(exp))
    return(ret_val)
def cleanse_labels(str_labels):
    try:
        str_labels = str_labels.replace('"choices":', "").replace('{ ["','').replace('"]}','').replace('"', '').replace(", ", ";")
    except:
        pass
    return(str_labels)

def get_name_of_file(x):
    return(Path(x).name)


def train(args):
    # parameter setting    
    ARGS_MODEL_DIR = args.model_dir
    ARGS_LEARNING_RATE = args.lr
    ARGS_EPOCHS = args.epochs
    print( "***", ARGS_MODEL_DIR, ARGS_LEARNING_RATE, ARGS_EPOCHS, type(ARGS_LEARNING_RATE), type(ARGS_EPOCHS), "***")
    SM_CHANNEL_TRAINING = os.environ['SM_CHANNEL_TRAINING']  # /opt/ml/input/data/training
    data_dir = os.environ['SM_CHANNEL_TRAINING']            # /opt/ml/input/data/training
    SM_MODEL_DIR = os.environ['SM_MODEL_DIR'] #args.model_dir        # /opt/ml/model
    SM_OUTPUT_DATA_DIR = os.environ['SM_OUTPUT_DATA_DIR'] # /opt/ml/output/data
    SM_CHANNEL_MODEL = os.environ['SM_CHANNEL_MODEL'] # /opt/ml/input/data/model
    IMAGE_DIR = SM_CHANNEL_TRAINING + "/images/"
    FASTAI_MODEL_DIR    =  SM_CHANNEL_TRAINING + "/models/"
    SM_OUTPUT_DIR = os.environ['SM_OUTPUT_DIR']
    SM_HPS = os.environ['SM_HPS'] #{"epochs": 3,"lr": 0.001 }

    
#     # Not deleting the following commented code as this helps in debugging in local mode.
#     SM_CHANNEL_TRAINING =  "./opt/ml/input/data/training"
#     data_dir            = "./opt/ml/input/data/training"
#     SM_MODEL_DIR        = './opt/ml/model'
#     SM_OUTPUT_DATA_DIR  = './opt/ml/output/data'
#     SM_CHANNEL_MODEL    = './opt/ml/input/data/model'
#     IMAGE_DIR           =  SM_CHANNEL_TRAINING + "/images/"
#     FASTAI_MODEL_DIR    =  SM_CHANNEL_TRAINING + "/models/"
#     SM_HPS ={"epochs": 2,"lr": 0.003 }
    
    
    # start 
    input_model_name = ""
    input_model_path = ""
    load_model_string = ""
    
    if len(os.listdir(SM_CHANNEL_MODEL)) > 0: # We have potentially found a model
        model_file_name = ""
        for model_file in os.listdir(SM_CHANNEL_MODEL):
            if model_file.find(".pth") > 0:
                input_model_name = model_file
        #input_model_name = os.listdir(SM_CHANNEL_MODEL)[0]
        print("We found a model in", SM_CHANNEL_MODEL, input_model_name)
        input_model_path = SM_CHANNEL_MODEL + "/" + input_model_name

    if (input_model_name != ""):
        load_model_string = input_model_name.replace(".pth", "")
        if not os.path.exists(FASTAI_MODEL_DIR):
            os.makedirs(FASTAI_MODEL_DIR) # Creating to keep the earlier model which is downloaded.
        shutil.copy(input_model_path, FASTAI_MODEL_DIR + input_model_name)
      
    print("1.", SM_CHANNEL_TRAINING, "2.", SM_CHANNEL_MODEL, "3.", SM_OUTPUT_DATA_DIR, "4", SM_MODEL_DIR, "\n")

    if not (os.path.exists(SM_CHANNEL_TRAINING + "/annotation-info.csv")):
        print(f"ERROR: Annotation file ( {SM_CHANNEL_TRAINING}/annotation-info.csv ) is not available, exiting ")

    annotation_file_path = f'{SM_CHANNEL_TRAINING}/annotation-info.csv'
    df_labels = pd.read_csv(annotation_file_path); df_labels.head()
    df_labels['choice'].value_counts()
    df_labels['labels'] = list(map(cleanse_labels, df_labels['choice']))
    df_labels['labels'].value_counts()
    df_training_labels =  df_labels[df_labels['labels'].isin(['Normal','Others','Myocardial Infarction'])].reset_index()
    df_training_labels['labels'].value_counts()
    df_training_labels['image_name'] = list(map(get_name_of_file, df_training_labels['image']))
    df_train = df_training_labels[['image_name', 'labels']]
    print("annotation file has", df_train.shape[0], " records.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train", "device", device)
    
    
    dls = ImageDataLoaders.from_df(df_train
                               , path = SM_CHANNEL_TRAINING
                               , folder = "images"
                               , label_col = 'labels'
                               , item_tfms = RandomResizedCrop(128, min_scale=0.35)
                               , bs = 32)
    
   
    
    print("DATALOADERS DONE!")
    #dls.show_batch()
    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.add_cb(CSVLogger())
    if(load_model_string != ""):
    # We have a .pth file here so transfer learning can be performed. Model is available in SM_MODEL_DIR
        try:
            learn = learn.load(load_model_string)
            print(load_model_string, "loaded for TRANSFER LEARNING.")
        except:
            print("loading of model failed", load_model_string)
    #df_curr = pd.read_csv('./history.csv')
    learn.unfreeze()
    print(f"Starting model training with epochs {ARGS_EPOCHS}, learning rate {ARGS_LEARNING_RATE}")
    learn.fit_one_cycle(ARGS_EPOCHS, ARGS_LEARNING_RATE)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"ecg-model-resnet34-{timestr}"
    try:
        shutil.copy(SM_CHANNEL_TRAINING + "/history.csv", SM_MODEL_DIR + "/" + model_name + ".csv" )
        shutil.copy(SM_CHANNEL_TRAINING + "/history.csv", SM_OUTPUT_DATA_DIR + "/" + model_name + ".csv" ) 
    except Exception as exp:
        print('exception in copying csv file.: ', str(exp))
    print("looking to save model string:", model_name)
    saved_model = learn.save(model_name)
    learn.remove_cb(CSVLogger)
    print(learn.path.ls())
    #learn.remove_cb(CSVLogger())
    learn.export(model_name + ".pkl") # This will be required for inferencing.
    print("SAVED MODEL FILES, BOTH!", saved_model)
    tmp_DEST_DIR = SM_MODEL_DIR
    tmp_SRC_DIR = str(learn.path)  +  "/" + learn.model_dir
    tmp_SRC_FILE = model_name + ".pth"
    #copy_generated_model(tmp_SRC_DIR, tmp_SRC_FILE, tmp_DEST_DIR)
    copy_generated_model(tmp_SRC_DIR, tmp_SRC_FILE, SM_MODEL_DIR)
    print("COPIED:", tmp_SRC_FILE, " to ", tmp_DEST_DIR )
    # copying .pkl model
    tmp_DEST_DIR = SM_MODEL_DIR
    tmp_SRC_DIR = SM_CHANNEL_TRAINING  +  "/" #+ learn.model_dir
    tmp_SRC_FILE = model_name + ".pkl"
    copy_generated_model(tmp_SRC_DIR, tmp_SRC_FILE, SM_MODEL_DIR)
    print("COPIED:", tmp_SRC_FILE, " to ", tmp_DEST_DIR )
    print("CHECKING model folder:", SM_MODEL_DIR, os.listdir(SM_MODEL_DIR))
    print("CHECKING model folder:", SM_OUTPUT_DATA_DIR, os.listdir(SM_OUTPUT_DATA_DIR))
    print("END OF TRAINING.")
    return(model_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    #parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     # Container environment
#     parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
#     parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
#     parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--model-channel', type=str, default=os.environ['SM_CHANNEL_MODEL'])
    model_identifier = train(parser.parse_args())
    #DELETED model_identifier = train(parser)
    print("TRAINING COMPLETED:MODEL-ID:", model_identifier)
    # SM_CHANNEL_MODEL=/opt/ml/input/data/model
    # SM_CHANNEL_TRAINING=/opt/ml/input/data/training