import scipy.io
import torch
import numpy as np
import os
import numpy.linalg as lng
import shutil
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax




def make_sudo_dataset(target, q_c, q_l, q_s_l,g_c,index):
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







def compute_mAP(index, qc,gallery_cam, good_index, junk_index):
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



def find_neighbors(qfr,qc,ql,fr_interval, data_frame,data_cam,data_label):
    
    neighbor_index =np.where((abs((np.array(data_frame) - qfr))< fr_interval)&(np.array(data_cam)==qc))[0]#&(np.array(data_label)!=ql))[0]
    # 
    return neighbor_index

def select_gallery_indexes_with_different_cam(gallery_cam, cam ):
    gallery_cam = np.array(gallery_cam)
    gallery_index = np.where(gallery_cam!=cam)[0]
    
    return gallery_index
 




def calculate_score_hungarian_without_redundancy(query_index,threshold,query_label, gallery_label,gallery_cam, neighbor_query_list,neighbor_gallery_list, query_gallery_c,number_of_random_samples ):

    

    # remove redundant samples from  gallery set first. We do not touch query as in market and duke query is always one image

    g_label_list = np.zeros(len(neighbor_gallery_list))
    g_cam_list = np.zeros(len(neighbor_gallery_list))
    i = 0
    for g in neighbor_gallery_list:
        g_label_list[i] = gallery_label[g]
        g_cam_list[i] = gallery_cam[g]
        i = i+1


    ngl = np.array(neighbor_gallery_list)
    gallery_new_list = []
    
   
    q = query_index
    
    j=0
    for g in neighbor_gallery_list:
        a = np.argmax(query_gallery_c[q,list(ngl[[list(np.where((g_label_list==g_label_list[j])&(g_cam_list==g_cam_list[j]))[0])][0]])])########33
        gallery_new_list.append( list(ngl[[list(np.where((g_label_list==g_label_list[j])&(g_cam_list==g_cam_list[j]))[0])][0]])[a])
        j = j+1



    gallery_new_list = set(gallery_new_list)
    gallery_new_list = list(gallery_new_list)

    gallery_new_list_cam = {"cam_1":[], "cam_2":[],"cam_3":[],"cam_4":[],"cam_5":[],"cam_6":[],"cam_7":[],"cam_8":[]}
    cost_cam = {"cam_1":[],"cam_2":[],"cam_3":[],"cam_4":[],"cam_5":[],"cam_6":[],"cam_7":[],"cam_8":[]}

    for i in range(8):
        gallery_new_list_cam["cam_"+str(i+1)] = np.array(gallery_new_list)[np.where(np.array(gallery_cam)[gallery_new_list]==i+1)[0]] 
        cost_cam["cam_"+str(i+1)] = np.zeros((len(neighbor_query_list),len(neighbor_query_list)+len(gallery_new_list_cam["cam_"+str(i+1)])))
    
    
    labels_present_in_query = np.array(query_label)[np.array(neighbor_query_list)]
    
    for cam in range(8):
        index = 0
        for q in neighbor_query_list:
            cost_cam["cam_"+str(cam+1)][index,0:len(gallery_new_list_cam["cam_"+str(cam+1)])] = query_gallery_c[q,gallery_new_list_cam["cam_"+str(cam+1)]]/(1-query_gallery_c[q,gallery_new_list_cam["cam_"+str(cam+1)]])
            index = index + 1

    cost_cam_naive = cost_cam.copy()
    



    for cam in range(8):  
        if len(gallery_new_list_cam["cam_"+str(cam+1)])>0:
            if np.where(cost_cam["cam_"+str(cam+1)]>0):
                # print(cost_cam["cam_"+str(cam+1)])
                cost_cam["cam_"+str(cam+1)] = estimate_probability(cost_cam["cam_"+str(cam+1)][:,0:len(gallery_new_list_cam["cam_"+str(cam+1)])],number_of_random_samples)
                # cost_cam["cam_"+str(cam+1)] = estimate_prob_for_cost_matrix(cost_cam["cam_"+str(cam+1)][:,0:len(gallery_new_list_cam["cam_"+str(cam+1)])])
        cost_cam["cam_"+str(cam+1)][:,len(gallery_new_list_cam["cam_"+str(cam+1)]):(len(neighbor_query_list)+len(gallery_new_list_cam["cam_"+str(cam+1)]))] = threshold



    #### comment ot of you estimate the cost

    labels_present_in_query = np.array(query_label)[np.array(neighbor_query_list)]
    







    return cost_cam,cost_cam_naive,gallery_new_list_cam,labels_present_in_query





def estimate_probability(cost,number_of_random_samples):

    # print(cost)
    


    row_ind, col_ind = linear_sum_assignment(-cost)
    data_1 = []
    data_2 = []
    row = len(cost)
    col = len(cost[0])


    new_cost = np.zeros([row,col]) 


    for i in range(len(row_ind)):
        for j in range(col):
            T= torch.tensor(1.0, requires_grad = True)
            sigma = torch.tensor(0.35, requires_grad = False)
           
            data_1 = []
            data_2 = []
            lab = []

            
            list_of_negative = list(cost[i,:])+list(cost[:,j])
            list_of_negative.remove(cost[i,j]) 
            list_of_negative.remove(cost[i,j])

            samples = np.random.normal(0, 1, 100)
          
     
            if len(list_of_negative) == 0:
                new_cost[i,j] = cost[i,j]
            
            else:
            

                mean_value = np.mean(np.array(list_of_negative))
                std = np.std(np.array(list_of_negative))
                max_val = np.max(np.array(list_of_negative))
                list_of_negative = np.random.normal(mean_value, std, 150)
                list_of_negative[list_of_negative<0] = 0.001
                list_of_negative = list(list_of_negative)

                label = list(np.ones(len(list_of_negative),dtype=int))
               
                data_1 = data_1 + list_of_negative
                
                pos = np.repeat(mean_value, len(data_1))

                data_2 = data_2 + list(pos)

                data_1 = data_1 + list(np.repeat(0, len(samples)))
                data_2 = data_2 + list(np.repeat(mean_value, len(samples)))
                

                label = label + list(np.zeros(len(samples),dtype=int))


           
                data = np.zeros([len(data_1),2])
                data[:,0] = data_1
                data[:,1] = data_2 
              
                
                data = torch.tensor(data)
                target = torch.tensor(label)
                target = target.type(torch.LongTensor) 



                for epoch in range(25):
                    
                    optim = torch.optim.SGD([T],  lr=0.1, momentum=0.2)

                    s = np.zeros([len(data[:,0]),2])
                    s = torch.tensor(s)
                    s[len(list_of_negative):len(list_of_negative)+len(samples),0] = torch.tensor(samples)*sigma+cost[i,j]
                    x = data+s
                    x[x<0] = 0.001

              
                    x = x/T


                    loss = torch.nn.CrossEntropyLoss()
                    output = loss(x,target)
                    
                    optim.zero_grad()

                   
                    output.backward()

                   
                    optim.step()
                    if sigma.item()<0:
                        sigma = torch.tensor(0.001, requires_grad = True)

                    if T.item()<0:
                        T = torch.tensor(0.001, requires_grad = True)


                   
               
                data[:,0] = 0

                mean_value = np.mean(np.array(list_of_negative))
                data[:,1] = mean_value
                s = np.zeros([len(data[:,0]),2])
                # s = torch.tensor(s)
                s[len(list_of_negative):len(list_of_negative)+len(samples),0] = samples*sigma.item()+cost[i,j]
                x = data+s
                x = x/T.item()
                
                


                a = softmax([cost[i,j]/T.item(),mean_value/T.item()])
                new_cost[i,j] = a[0]



    return new_cost   









    




# def find_gallery_index_probably_matched(score,top,cam,gallery_cam,f_score,t_score):

#     index = []
#     idx = np.argsort(-score)
#     c = gallery_cam[idx[0]]
#     # index.append(idx[0])
#     for i in range(8):
#         if len(np.where(np.array(gallery_cam)[idx]==i+1)[0])>0:
#             index.append(idx[np.where(np.array(gallery_cam)[idx]==i+1)[0][0]])
    
#     return idx[0:top], index, f_score[idx[0]],t_score[idx[0]]


def select_gallery_index_from_specific_camview(gallery_cam,cam):

    return np.where(gallery_cam==cam+1)[0]




