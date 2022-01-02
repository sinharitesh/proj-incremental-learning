#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install fastai==2.3.1 > /dev/null')
get_ipython().system(' pip install pillow==8.2 > /dev/null')
import fastai
import PIL
print(fastai.__version__, PIL.__version__)


# In[2]:


import pandas as pd
import boto3
import os
from pathlib import Path
import re
import sagemaker
from sagemaker import get_execution_role
import zipfile
import shutil
import uuid
import time
from sagemaker.pytorch import PyTorch
from fastai.vision.all import *
import torch
s3 = boto3.client("s3")


# In[3]:


def get_base_jobname(prefix = "ecg-dl"):
    ret = str(uuid.uuid4())
    ret =ret[0:8].replace("=", "a")
    return(prefix + "-" +  ret)


# In[4]:


#PREV_MODEL_NAME_PTH = 'ecg-model-resnet34-20211230-173706.pth' # This is the model which will be loaded for transfer learning.
PREV_MODEL_NAME_PTH = 'ecg-model-resnet34-20220101-145613.pth' # This is the model which will be loaded for transfer learning.


# In[5]:


PREVIOUS_LANG_MODEL_PATH = "s3://ecg-models/"
TMP_MODEL_OUTPUT_URI = "s3://tmp-model-artefacts"
FINAL_BUCKET = "ecg-models"
MODEL_BUCKET = "ecg-models"
MODEL_PREFIX = "ecg"
ECG_DATA_BUCKET_S3 = "ritesh-s3-ecg-annotated-data"
PREVIOUS_LANG_MODEL_URI = f"s3://{MODEL_BUCKET}/{PREV_MODEL_NAME_PTH}" # This is pytorch model's S3 address 
# Above model will be downloaded by sagemaker train job to perform incremental learning.


# In[6]:


# RETAIN : Not required but useful code to list all the objects in a bucket #
# all_models = s3.list_objects( Bucket = FINAL_BUCKET) ; 
# lst_available_models = []
# for dct_model_info in all_models['Contents']:
#     str_key = dct_model_info["Key"]
#     if (str_key.find("encoder") == -1):
#         str_key = lst_available_models.append(str_key)
# lst_available_models.sort(reverse= True)


# In[7]:


# Data Prep Do not need to do every time, should only be done when new data is there
# try:
#     shutil.rmtree('./annotated-data')
# except:
#     pass
# os.makedirs('./annotated-data')
# os.makedirs('./annotated-data/images')
# #Downloading data from s3 bucket which have annotations available.
# ! cp ./annotation-info.csv  ./ecg-annotated-data/annotation-info.csv

# ECG_DATA_BUCKET = "ritesh-s3-ecg-annotated-data"
# s3 = boto3.client("s3")
# for fl in lst_images:
#     fn = Path(fl).name
#     #print(fn)
#     s3.download_file(ECG_DATA_BUCKET,fn,f'./ecg-annotated-data/images/{fn}')


# In[8]:


sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/ecg-model-data'
role = sagemaker.get_execution_role()
inputs = ""


# In[9]:


# If data is already in the inputs location through an earlier run, then just provide the path, nothing to do here. 
# Following command will upload the data to S3 bucket
inputs = "s3://sagemaker-us-east-2-096374906812/sagemaker/ecg-model-data"
bucket = "sagemaker-us-east-2-096374906812"
#inputs = sagemaker_session.upload_data( path = "./ecg-annotated-data/", bucket=bucket, key_prefix=prefix)
#print('input spec (in this case, just an S3 path): {}'.format(inputs))
# if (inputs == ""):
#     inputs = "s3://sagemaker-us-east-2-096374906812/sagemaker/ecg-model-data"
print(bucket, inputs)


# In[10]:


base_job_name = get_base_jobname()
hyperparameters={"epochs": 5, "lr": 3e-3}
use_spot_instances = False
max_run = 900
max_wait = 1000 if use_spot_instances else None
instance_type = 'ml.m5.xlarge' #"ml.p2.xlarge", # ml.p2.xlarge # ml.c5.2xlarge
checkpoint_s3_uri = (
    "s3://{}/{}/checkpoints/{}".format(bucket, prefix, job_name) if use_spot_instances else None
)


# Printing Training Inputs:
print("************************************ Starting Training: ***************************************")
print("Data Bucket:", prefix)
print("Latest Model Uri:", PREVIOUS_LANG_MODEL_URI )
print("Temp. model Output :", TMP_MODEL_OUTPUT_URI )
print("Base job name:", base_job_name)
print("checkpoint_s3_uri:", checkpoint_s3_uri)
print("************************************ ****************** ***************************************")

env = {
    'SAGEMAKER_REQUIREMENTS': 'requirements.txt', # path relative to `source_dir` below.
}

estimator = PyTorch(
    entry_point="classification-training.py",
    role=role,
    base_job_name  = base_job_name,
    framework_version="1.8",
    instance_count=1,
    instance_type= instance_type,
    source_dir=".",
    py_version="py3",
    env = env,
    model_uri = PREVIOUS_LANG_MODEL_URI,
    output_path = TMP_MODEL_OUTPUT_URI,
    hyperparameters = hyperparameters,
    checkpoint_s3_uri=checkpoint_s3_uri,
    use_spot_instances=use_spot_instances,
    max_run=max_run,
    max_wait=max_wait
    # git_config=git_config,
    # available hyperparameters: emsize, nhid, nlayers, lr, clip, epochs, batch_size,
    # bptt, dropout, tied, seed, log_interval
)
print("Invoking training job now.....with inputs", inputs)
estimator.fit(inputs)
training_job_name = estimator.latest_training_job.name
print("TRAINING JOB NAME:", training_job_name)


# In[11]:


# The job above uploads the models into output_path which is a S3 bucket. 
# The training_job_name is a folder under which output dirctory contains the models in a file called model.tar.gz.


# In[12]:


key = training_job_name + "/output/model.tar.gz" ; print(key)


# In[13]:


# The output of training job is saved in the bucket specified. 
# This is the place where files are extracted and saved in a specified location for further usage.
# then upload above files to the destination S3 directory.
# Finally cleaning up the temporary bucket where estimator has uploaded the model.
if not os.path.exists('./tmp-data'):
    os.makedirs("./tmp-data")
s3.download_file('tmp-model-artefacts', key, "./tmp-data/model.tar.gz")  
file_extract('./tmp-data/model.tar.gz', './tmp-data/extract/')
upload_files = Path('./tmp-data/extract/').ls()
for fl in upload_files:
    print("uploading: ", fl, " to: ", FINAL_BUCKET)
    s3.upload_file(str(fl), FINAL_BUCKET, str(fl.name))
#cleanup
try:
    shutil.rmtree("./tmp-data")
except:
    pass


# In[17]:


print(training_job_name)
s3.delete_object(Bucket= 'tmp-model-artefacts', Key= training_job_name)
#print(response)


# In[19]:


#! aws s3 rm s3://tmp-model-artefacts/{training_job_name} --recursive --dryrun


# In[20]:


#! aws s3 rm s3://tmp-model-artefacts/{training_job_name} --recursive

