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
    data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'_s.mat'
else:
    data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'.mat'

if opt.S:
    cosine_dir ='./rep/'+model_name+'/'+opt.source+"_"+opt.target+'_cosine_score_'+opt.query_type+'_s.mat'
else:
    cosine_dir ='./rep/'+model_name+'/'+opt.source+"_"+opt.target+'_cosine_score_'+opt.query_type+'.mat'







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
cosine_scores = scipy.io.loadmat(cosine_dir)
query_gallery_c = cosine_scores['query_gallery_score']
gallery_gallery_c = cosine_scores['gallery_gallery_score']  




###############################################          Compute RE-RANK SCORE ####################################
query_gallery_rerank = scores().rerank_score(query_gallery_c,gallery_gallery_c,top=10)


###############################################           Compute TEMPORAL SCORE   ################################

time_dictionary= scores().generate_sudo_label( query_label,query_frame, query_cam, gallery_frame,gallery_cam, query_gallery_c)
time_distribution = scores().estimate_time_distribution(time_dictionary)

###############################################             INITIALIZATION    ##################################################
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0        
count = 0
conventional_label = np.zeros(len(query_label))
temp_score =[]
temp_label = []


hung_no_redundant_label = np.zeros(len(query_label))
hung_no_redundant_label_naive = np.zeros(len(query_label))
c_final = np.zeros(len(query_label))
c_final_naive = np.zeros(len(query_label))
conv_final = np.zeros(len(query_label))
cam_label = np.zeros(len(query_label))


# # # #################################################     MAX Score (Conventional method)        ###########################################



for q in range(len(query_label)):      


    ap_tmp, CMC_tmp,_,_,_ = scores().score_calculation(query_gallery_c[q,:],query_gallery_rerank[q,:],query_cam[q],query_frame[q],query_label[q],gallery_cam,gallery_frame,gallery_label,time_distribution,cam = 1)

    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp




# #       ############################################    Hungarian without redundancy   ##############################################





    neighbor_idx_query = find_neighbors(query_frame[q],query_cam[q],query_label[q],neighbor_interval,query_frame,query_cam,query_label)
    neighbor_idx_query = list(neighbor_idx_query)
    neighbor_idx_query.remove(q)
    neighbor_idx_query.insert(0,q)
    neighbor_idx_query = np.array(neighbor_idx_query)  
    neighbor_idx_gallery = []

    # for each neighbor samples in the query list, we will find the best match in the gallery
    
    

    
    for n in neighbor_idx_query:    
        _, _,index,_,cam_index = scores().score_calculation(query_gallery_c[n,:],query_gallery_rerank[n,:],query_cam[n],query_frame[n],query_label[n], gallery_cam,gallery_frame,gallery_label,time_distribution,cam=1)
        neighbor_idx_gallery = np.append(neighbor_idx_gallery,find_neighbors(gallery_frame[index],gallery_cam[index],gallery_label[index],neighbor_interval, gallery_frame,gallery_cam,gallery_label))
   
    neighbor_idx_gallery = np.array(neighbor_idx_gallery)
    neighbor_idx_gallery = list(neighbor_idx_gallery)
    neighbor_idx_gallery = set(neighbor_idx_gallery)
    neighbor_idx_gallery = list(neighbor_idx_gallery)   
    neighbor_idx_gallery = list(map(int, neighbor_idx_gallery))

    

    # # we have query neighbors and gallery neighbors for one camera, we apply Hungarian
    cost_cam,cost_cam_naive,gallery_new_list_cam,query_list= calculate_score_hungarian_without_redundancy(q, 0, query_label, gallery_label,gallery_cam, neighbor_idx_query,neighbor_idx_gallery, query_gallery_c, 100)

    col_index = {"cam_1":[],"cam_2":[],"cam_3":[],"cam_4":[],"cam_5":[],"cam_6":[],"cam_7":[],"cam_8":[]}
    col_index_naive = {"cam_1":[],"cam_2":[],"cam_3":[],"cam_4":[],"cam_5":[],"cam_6":[],"cam_7":[],"cam_8":[]}
    for i in range(8):
        print("cam",str(i+1),cost_cam["cam_"+str(i+1)])
        row_ind, col_ind = linear_sum_assignment(-cost_cam["cam_"+str(i+1)])
        col_index["cam_"+str(i+1)] = col_ind
        row_ind_navie, col_ind_naive = linear_sum_assignment(-cost_cam_naive["cam_"+str(i+1)])
        col_index_naive["cam_"+str(i+1)] = col_ind_naive



    query_idx_in_query_list = int(np.where(np.array(neighbor_idx_query)==q)[0][0])
    gidx = np.zeros(8)
    gidx_naive = np.zeros(8)
    c = np.zeros(8)
    c_n = np.zeros(8)


    for i in range(8):

        gidx[i] = int(col_index["cam_"+str(i+1)][query_idx_in_query_list])
        gidx_naive[i] = int(col_index_naive["cam_"+str(i+1)][query_idx_in_query_list])
        c[i] = cost_cam["cam_"+str(i+1)][query_idx_in_query_list,int(gidx[i])]
        c_n[i] = cost_cam_naive["cam_"+str(i+1)][query_idx_in_query_list,int(gidx_naive[i])]

    


    gidx =  np.array(gidx, dtype='int')
    gidx_naive =  np.array(gidx_naive, dtype='int')
    
    index_max = np.argmax(c)
    hung_no_redundant_label[q] = -10

    index_max_naive = np.argmax(c_n)
    hung_no_redundant_label_naive[q] = -10

    result = [-1,-1,-1,-1,-1,-1,-1,-1]
    if (c[index_max] < 0) :
        hung_no_redundant_label[q] = -10
    


    
    else:
        for i in range(8):
            if (gidx[i]<len(gallery_new_list_cam["cam_"+str(i+1)])):
                result[i] = gallery_label[gallery_new_list_cam["cam_"+str(i+1)][gidx[i]]]
            
            if (index_max == i) and (gidx[index_max]<len(gallery_new_list_cam["cam_"+str(i+1)])):
                hung_no_redundant_label[q] = gallery_label[gallery_new_list_cam["cam_"+str(i+1)][gidx[index_max]]]
                cam_label[q] = i+1
            


    if (c_n[index_max] <0) :
        hung_no_redundant_label_naive[q] = -10
    else:
        for i in range(8):
            if (index_max_naive == i) and (gidx_naive[index_max_naive]<len(gallery_new_list_cam["cam_"+str(i+1)])):
                hung_no_redundant_label_naive[q] = gallery_label[gallery_new_list_cam["cam_"+str(i+1)][gidx_naive[index_max_naive]]]


    print(query_label[q],hung_no_redundant_label[q] )
    print(query_label[q],hung_no_redundant_label_naive[q])
        




###################################   MAX RESULT   ################################
CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))



#####################################  HUNGARIAN RESULT ##################################
print("accuracy with adjusted probability",len(np.where(hung_no_redundant_label_naive==query_label)[0])/len(query_label))
print("accuracy with simple cost", len(np.where(hung_no_redundant_label==query_label)[0])/len(query_label))