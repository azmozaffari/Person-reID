import numpy as np
import numpy.linalg as lng
import math
from utilities import select_gallery_indexes_with_different_cam, compute_mAP
from sklearn.preprocessing import normalize

class scores():
    
    
    def calculate_re_rank_respecting_to_gallery(self, gallery_gallery_score,top):
        gallery_re_rank_score = np.zeros((gallery_gallery_score.shape[0],top))
        gallery_score_index = np.zeros((gallery_gallery_score.shape[0],gallery_gallery_score.shape[1]))
        for i in range(gallery_gallery_score.shape[0]):
            gallery_re_rank_score[i,:] = np.argsort(-gallery_gallery_score[i,:])[0:top]
            for j in range(top):
                gallery_score_index[i,int(gallery_re_rank_score[i,j])]=1 
        return gallery_score_index
 



    def rerank_score(self, query_gallery_cosine,gallery_gallery_cosine,top):    
        gallery_score_index =  self.calculate_re_rank_respecting_to_gallery(gallery_gallery_cosine,top=10)
        query_score_index = self.calculate_re_rank_respecting_to_gallery(query_gallery_cosine,top=10)
        return ((1/top)*np.dot(query_score_index,np.transpose(gallery_score_index)))
 


    def cosine_score(self, query_feature,gallery_feature):        
        query_f = normalize(query_feature, axis=1, norm='l2')
        gallery_f = normalize(gallery_feature, axis=1, norm='l2')
        score = np.matmul(query_f, np.transpose(gallery_f))
        return score


   

    def generate_sudo_label(self, query_label,query_frames, query_cam, gallery_frames,gallery_cam, query_gallery_c):

    
        key_list = []
        for i in range(8-1):
            for j in range(i+1,8):
                key_list.append(str(i+1)+'_'+str(j+1))

        time_dictionary = dict.fromkeys(key_list)
        feature_dictionary = dict.fromkeys(key_list)

        index_query = 0

        for q in range(len(query_label)):
       
            i=0
            
            index = np.argsort(-query_gallery_c[q,:])
            for j in range(20):
                while (gallery_cam[index[i]]==query_cam[index_query]) & (i <len(gallery_cam)):
                    i+=1
                if (gallery_cam[index[i]]!=query_cam[index_query]):
                    
                    if query_cam[index_query]<gallery_cam[index[i]]:
                        key = str(query_cam[index_query])+'_'+str(gallery_cam[index[i]])
                    else:
                        key = str(gallery_cam[index[i]])+'_'+str(query_cam[index_query])

                    temp = []
                    temp_f = []

                    if time_dictionary[key] != None:
                        temp = np.array(time_dictionary[key])
                        temp = list(temp)
                        a = abs(query_frames[index_query] - gallery_frames[index[i]])
                        temp.append(str(a))



                    else:
                        a = (abs(query_frames[index_query] - gallery_frames[index[i]]))
                        temp.append(str(a))

                        
                    time_dictionary[key] = temp
                  
            
            index_query +=1

        return time_dictionary






    def estimate_time_distribution(self, time_dictionary):

        x = time_dictionary
        max_time_difference = 0
        min_time_difference = 0
        for i in range(8-1):
            for j in range(i+1,8):
                if x[str(i+1)+'_'+str(j+1)] == None:
                    x[str(i+1)+'_'+str(j+1)] = '0'
                # test_list = list(map(int, x[str(i+1)+'_'+str(j+1)]))
                test_list = [math.floor(float(k)) for k in x[str(i+1)+'_'+str(j+1)]]
                # ist(map(lambda x: math.floor(float(x)), i))
                # test_list = list(map(int, math.floor(x[str(i+1)+'_'+str(j+1)])))
                
                if np.max(np.array(test_list))>max_time_difference:
                    max_time_difference = np.max(np.array(test_list))
                if np.min(np.array(test_list))<min_time_difference:
                    min_time_difference = np.min(np.array(test_list))


        step = 5000
        distribution_time = np.zeros((8,8,int(max_time_difference/step)+2))
        for i in range(8-1):
            for j in range(i+1,8):          
                if x[str(i+1)+'_'+str(j+1)] != '0':
                    # test_list = list(map(int, math.floor(x[str(i+1)+'_'+str(j+1)])))
                    test_list = [math.floor(float(k)) for k in x[str(i+1)+'_'+str(j+1)]]
                
                    for k in test_list:
                        distribution_time[i,j,int(np.array(k)/step)] +=1 

   
        return distribution_time




    def score_calculation(self,score_cosine,rerank,qc,qfr,ql,gc,gfr,gl,time_distribution,cam):
    

        gallery_time = np.array(gfr)
        time_difference = abs(np.array(qfr)-gallery_time)
        time_probability = np.zeros((gc.shape[0]))

        gallery_index = 0
        interval = 5000 

        d3 = time_distribution.shape[2]


        for g in gc:
            if np.array(qc)<np.array(g):
                
                if int(time_difference[ gallery_index]/interval)>d3-1:
                    time_probability[gallery_index] = time_distribution[qc-1,g-1,-1]/(sum(time_distribution[qc-1,g-1,:])+1e-10)
                else:
                    time_probability[gallery_index] = time_distribution[qc-1,g-1,int(time_difference[gallery_index]/interval)]/(sum(time_distribution[qc-1,g-1,:])+1e-10)      
            else:
                if int(time_difference[ gallery_index]/interval)>d3-1:
                    time_probability[gallery_index] = time_distribution[g-1,qc-1,-1]/(sum(time_distribution[g-1,qc-1,:])+1e-10)

                else:
       
                    time_probability[gallery_index] = time_distribution[g-1,qc-1,int(time_difference[gallery_index]/interval)]/(sum(time_distribution[g-1,qc-1,:])+1e-10)

            gallery_index+=1


        # score = (score_cosine)
        # score = (time_probability+0.005)
        # score = (rerank+0.1)
        # score = np.multiply((score_cosine),(rerank+0.1))
        # score = np.multiply( (time_probability+0.005),(rerank+0.1))
        # score = np.multiply( (score_cosine),(time_probability+0.005))
                        

        score = np.multiply(np.multiply((score_cosine), (time_probability+0.005)),rerank+0.1)
        
        index = np.argsort(-score)  #from small to large

        # for the given camera select the matched gallery samples
        cam_index = np.argwhere(np.array(gc) == cam)
        mask = np.in1d(index, cam_index)        
        gallery_selected_index_cam = index[mask]
        
        # good index
        query_index = np.argwhere(gl==ql)
        #same camera
        camera_index = np.argwhere(gc==qc)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl==-1)

        junk_index2 = camera_index
        junk_index = np.append(junk_index2, junk_index1)
        
        ap,CMC_tmp,gallery_selected_index = compute_mAP( index,qc,gc, good_index, junk_index)

        return ap, CMC_tmp,gallery_selected_index,score[gallery_selected_index],gallery_selected_index_cam[0]







    def simple_score_calculation(self,score,qc,qfr,ql,gc,gfr,gl):
    

        
        
        index = np.argsort(-score)  #from small to large

        # for the given camera select the matched gallery samples
        
        
        # good index
        query_index = np.argwhere(gl==ql)
        #same camera
        camera_index = np.argwhere(gc==qc)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl==-1)

        junk_index2 = camera_index
        junk_index = np.append(junk_index2, junk_index1)
        
        ap,CMC_tmp,gallery_selected_index = compute_mAP( index,qc,gc, good_index, junk_index)

        return ap, CMC_tmp

