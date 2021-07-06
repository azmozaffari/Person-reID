# 2     extract score and sudo label for every image in query tracklets and then save in a file
import argparse
import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy.linalg as lng
import shutil










parser = argparse.ArgumentParser(description='Evaluate')
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







#######################################################################
# Evaluate


cam_metric = torch.zeros(8,8)


def copy_files_in_sudo_query_folders(target, q_c, q_l, q_s_l,g_c,index):
#q_l is the query label, q_s_l is the given label to the query, g_c is the camera view of the matched gallery sample
    main_dir = "./dataset/modified_dataset"

    if index == 0:
        # make folder to save the data
        if os.path.exists(main_dir+"/"+target+"/"+"sudo_labeled"):
            shutil.rmtree(main_dir+"/"+target+"/"+"sudo_labeled")    
        os.mkdir(main_dir+"/"+target+"/"+"sudo_labeled")
        for c in range(1,9): # camera range
            os.mkdir(main_dir+"/"+target+"/"+"sudo_labeled"+"/"+str(c))

    

    if not os.path.exists(main_dir+"/"+opt.target+"/"+"sudo_labeled"+"/"+str(g_c)+"/"+str(q_s_l).zfill(4)):
        os.mkdir(main_dir+"/"+opt.target+"/"+"sudo_labeled"+"/"+str(g_c)+"/"+str(q_s_l).zfill(4))

    
    # copy query tracklet samples to the folder
    query_list = os.listdir(main_dir+"/"+opt.target+"/"+"multi_query"+"/"+str(q_l).zfill(4))

    for file in query_list:
        if (file[0:7] == str(q_l).zfill(4)+"_c"+str(q_c)):
            temp = (str(q_s_l).zfill(4))
            
            file_temp  = temp + (file[4:-1])
            
            shutil.copy(main_dir+"/"+opt.target+"/"+"multi_query"+"/"+str(q_l).zfill(4)+"/"+file,main_dir+"/"+opt.target+"/"+"sudo_labeled"+"/"+str(g_c)+"/"+str(q_s_l).zfill(4)+"/"+file_temp )

    
    # copy gallery tracklet images to the sudo label folder 
    gallery_list = os.listdir(main_dir+"/"+opt.target+"/"+"gallery"+"/"+str(q_s_l).zfill(4))
    for file in gallery_list:
        if (file[0:7] == str(q_s_l).zfill(4)+"_c"+str(g_c)):
            shutil.copy(main_dir+"/"+opt.target+"/"+"gallery"+"/"+str(q_s_l).zfill(4)+"/"+file,main_dir+"/"+opt.target+"/"+"sudo_labeled"+"/"+str(g_c)+"/"+str(q_s_l).zfill(4)+"/"+file )





def evaluate(score,qf,ql,qc,gf,gl,gc):
    # we only want to select the answer among the samples that are from different camera view


    # query = qf.view(-1,1)
    # # print(query.shape)
    # score = torch.mm(gf,query)
    # score = score.squeeze(1).cpu()
    # score = score.numpy()
    
        
    index = np.argsort(-score)  #from small to large
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)

    junk_index2 = camera_index
    junk_index = np.append(junk_index2, junk_index1)
    
    ap,CMC_tmp,gallery_selected_index = compute_mAP( index, qc, good_index, junk_index)

    return ap, CMC_tmp,score[gallery_selected_index], gl[gallery_selected_index], gallery_cam[gallery_selected_index],gallery_selected_index 


def compute_mAP(index, qc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(index, junk_index, invert=True)
    #mask2 = np.in1d(index, np.append(good_index,junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]
    for i in range(8):
        cam_metric[ qc-1, ranked_camera[i]-1 ] +=1

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc, index[0]

######################################################################
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


# print(query_feature.size, query_cam.shape, query_label.shape, query_frame.shape)

# compute cosine score 
c = np.matmul(query_feature,gallery_feature.T)
norm_query = lng.norm(query_feature,2, axis=1)
norm_gallery = lng.norm(gallery_feature,2, axis=1)

norm_matrix = np.matmul(np.expand_dims(np.array(norm_query),axis=1),np.expand_dims(np.array(norm_gallery),axis=1).T)
score = c/norm_matrix







query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()







CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

g_matched_frame = []
g_matched_camera = []
for i in range(len(query_label)):
    ap_tmp, CMC_tmp, s_q,q_s_l,g_c,g_index = evaluate(torch.tensor(score[i,:]),query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    g_matched_camera.append(g_c)
    g_matched_frame.append(gallery_frame[g_index])
    copy_files_in_sudo_query_folders(opt.target, query_cam[i], query_label[i], q_s_l,g_c, i)

    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

result = {'gallery_frame':g_matched_frame, 'gallery_cam':g_matched_camera,'query_frame':query_frame,'query_cam':query_cam}
scipy.io.savemat('./rep/'+'frame_cam_match_data.mat',result)

# query_feature = query_feature.cpu()
# gallery_feature = gallery_feature.cpu()





