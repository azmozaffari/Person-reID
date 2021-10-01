import os
from shutil import copyfile
import argparse
from os import listdir
import shutil
import numpy as np

dataset_name = 'duke'
dataset_folder = './dataset'
original_dataset_folder = dataset_folder+"/"+"original_dataset"+"/"+dataset_name
modified_dataset_folder = dataset_folder+"/"+"modified_dataset"+"/"+dataset_name

file_names_train = listdir(original_dataset_folder+"/"+"bounding_box_train")
file_names_gallery = listdir(original_dataset_folder+"/"+"bounding_box_test")
file_names_query = listdir(original_dataset_folder+"/"+"query")

print(len(file_names_gallery))



for file in file_names_query:
    

    ID = file[0:4]
    cam = file[6]
    
    
    phrase = ID+"_c"+cam

    # temp = file_names_gallery[:,0:7]

    # print(phrase, file_names_gallery[temp ==phrase])