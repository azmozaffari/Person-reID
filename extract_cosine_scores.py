import argparse
import scipy.io
import torch
import numpy as np
import os
import numpy.linalg as lng
import shutil
from scores import scores




parser = argparse.ArgumentParser(description='Cosine')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--source', default ='duke' , help='training domain' )
parser.add_argument('--target', default ='market' , help='test domain' )
parser.add_argument('--query_type', default ='query' , help='query, multi_query' )
parser.add_argument('--S', action='store_true' , help='for each tracklet it calculates mean ' )


opt = parser.parse_args()

if opt.PCB:
    model_name = opt.model_name+"_"+"pcb"+"_"+opt.source+"_e"
else:
    model_name = opt.model_name+"_"+opt.source+"_e"


str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
# name = opt.name

if opt.S:
    data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'_s.mat'
else:
    data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'.mat'




result = scipy.io.loadmat(data_dir)

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




q_g_score = scores().cosine_score(query_feature,gallery_feature)
g_g_score = scores().cosine_score(gallery_feature,gallery_feature)
g_q_score = scores().cosine_score(gallery_feature,query_feature)


data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'_s.mat'
result = {'query_gallery_score':np.array(q_g_score), 'gallery_gallery_score':np.array(g_g_score)}

if opt.S:

	scipy.io.savemat('./rep/'+model_name+'/'+opt.source+"_"+opt.target+'_cosine_score_'+opt.query_type+'_s.mat',result)
else:
	scipy.io.savemat('./rep/'+model_name+'/'+opt.source+"_"+opt.target+'_cosine_score_'+opt.query_type+'.mat',result)

