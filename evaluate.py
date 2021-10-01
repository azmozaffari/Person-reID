from __future__ import print_function, division
import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
import random
import copy
from sklearn.preprocessing import normalize
# from prettytable import PrettyTable
from numpy import linalg as LA   
from scipy.special import softmax 
import torch
import torch.nn.functional as F
import csv
from scores import scores
import scipy.io
from utilities import * #estimate_probability,select_gallery_index_from_specific_camview,find_neighbors, select_gallery_indexes_with_different_cam, calculate_score_hungarian_without_redundancy,find_gallery_index_probably_matched
from scipy.optimize import linear_sum_assignment
import math

####################################################    OPTIONS  ##########################################################
parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--source', default ='duke' , help='training domain' )
parser.add_argument('--target', default ='market' , help='test domain' )
parser.add_argument('--query_type', default ='query' , help='query, multi_query' )
parser.add_argument('--S', action='store_true' , help='for each tracklet it calculates mean ' )
parser.add_argument('--neighbor_interval',default= 1300, help='neighbor frame interval')

opt = parser.parse_args()


neighbor_interval = opt.neighbor_interval

if opt.PCB:
    model_name = opt.model_name+"_"+"pcb"+"_"+opt.source+"_e"
else:
    model_name = opt.model_name+"_"+opt.source+"_e"

str_ids = opt.gpu_ids.split(',')

if opt.S:
    data_dir = './rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'_s.mat'
else:
    data_dir = './rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'.mat'

if opt.S:
    cosine_dir ='./rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_cosine_score_'+opt.query_type+'_s.mat'
else:
    cosine_dir ='./rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_cosine_score_'+opt.query_type+'.mat'







# #############################################       LOAD DATA  #############################################
result = scipy.io.loadmat(data_dir)
data = result
if opt.S:
    query_feature = torch.FloatTensor(result['query_f']).cpu()
    query_cam = result['query_cam'][0]
    query_label = result['query_label']
    query_frame = result['query_frames'][0]

    gallery_feature = torch.FloatTensor(result['gallery_f']).cpu()
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label']
    gallery_frame = result['gallery_frames'][0]
else:
    query_feature = torch.FloatTensor(result['query_f']).cpu()
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    query_frame = result['query_frames'][0]

    gallery_feature = torch.FloatTensor(result['gallery_f']).cpu()
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
    gallery_frame = result['gallery_frames'][0]













###############################################          Compute COSINE SCORE ####################################
query_gallery_c = scores().cosine_score(query_feature,gallery_feature)



###############################################          Compute RE-RANK SCORE ####################################
# query_gallery_rerank = scores().rerank_score(query_gallery_c,gallery_gallery_c,top=10)


###############################################           Compute TEMPORAL SCORE   ################################

# time_dictionary= scores().generate_sudo_label( query_label,query_frame, query_cam, gallery_frame,gallery_cam, query_gallery_c)
# time_distribution = scores().estimate_time_distribution(time_dictionary)

###############################################             INITIALIZATION    ##################################################
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0        
count = 0
conventional_label = np.zeros(len(query_label))
temp_score =[]
temp_label = []
cost_adjusted = np.zeros(len(query_label))
cost_naive = np.zeros(len(query_label))

hung_no_redundant_label = np.zeros(len(query_label))
hung_no_redundant_label_naive = np.zeros(len(query_label))
c_final = np.zeros(len(query_label))
c_final_naive = np.zeros(len(query_label))
conv_final = np.zeros(len(query_label))
cam_label = np.zeros(len(query_label))


# # # #################################################     MAX Score (Conventional method)        ###########################################



for q in range(len(query_label)):      


    ap_tmp, CMC_tmp = scores().simple_score_calculation(query_gallery_c[q,:],query_cam[q],query_frame[q],query_label[q],gallery_cam,gallery_frame,gallery_label)

    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

###################################   MAX RESULT   ################################
CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Max results: Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
