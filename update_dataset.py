import scipy.io
import argparse
import numpy as np
import os
import shutil

parser = argparse.ArgumentParser(description='build up sudo dataset')
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

if opt.S:
    data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_final_result_'+opt.query_type+'_s'+'.mat'
else:
    data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_final_result_'+opt.query_type+'.mat'

# data_dir = './rep/'+model_name+'/'+opt.source+"_"+opt.target+'_max_result_cosinescore'+opt.query_type+'.mat'




#lambda
Threshold = 0.6

result_adjusted = scipy.io.loadmat(data_dir)

cost_adjusted = result_adjusted['cost'][0]
mlabel_adjusted = result_adjusted['matched_label'][0]
mcam_adjusted = result_adjusted['matched_cam'][0]

qlabel_adjusted = result_adjusted['query_label'][0]
qcam_adjusted = result_adjusted['query_cam'][0]


mask = np.where((cost_adjusted>Threshold))[0]
query_label = qlabel_adjusted[mask]
query_cam = qcam_adjusted[mask]
gallery_label = mlabel_adjusted[mask]
gallery_cam = mcam_adjusted[mask]



main_dir = "./dataset/modified_dataset/"+opt.target+"/sudo_label"
if os.path.exists(main_dir):
    shutil.rmtree(main_dir)

os.mkdir(main_dir)

for c in range(8):
    os.mkdir(main_dir+"/"+str(c+1))        


main_query_dir = "./dataset/modified_dataset/"+opt.target+"/multi_query"
main_gallery_dir = "./dataset/modified_dataset/"+opt.target+"/gallery"




# read all gallery files and make the new dataset with camera subfolders
for root, dirs, files in os.walk(main_gallery_dir):
    for file in files:
        f = file.split("_")
        

        if not os.path.exists(main_dir+"/"+str(f[1].split("c")[1])+"/"+f[0]):
            
            os.mkdir(main_dir+"/"+str(f[1].split("c")[1])+"/"+f[0])
        shutil.copy(main_gallery_dir+"/"+f[0]+"/"+file, main_dir+"/"+str(f[1].split("c")[1])+"/"+f[0]+"/"+file)









for q in range(len(query_label)):
    query_list = os.listdir(main_query_dir+"/"+str(query_label[q]).zfill(4))
    for list_q in query_list:
        if list_q[0:7] == str(int(query_label[q])).zfill(4)+"_c"+str(int(query_cam[q])):
            if gallery_label[q] == -1:
                if not os.path.exists(main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q]))):
                    os.mkdir(main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q])))
                # print(main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q]))+"/"+list_q)
                shutil.copy(main_query_dir+"/"+str(query_label[q]).zfill(4)+"/"+list_q,main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q]))+"/"+list_q)
                



            else:
                if not os.path.exists(main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q])).zfill(4)):
                    os.mkdir(main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q])).zfill(4))

                shutil.copy(main_query_dir+"/"+str(query_label[q]).zfill(4)+"/"+list_q,main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q])).zfill(4)+"/"+list_q)
                

    if gallery_label[q] == -1:
        gallery_list = os.listdir(main_gallery_dir+"/"+str(int(gallery_label[q])))
        
        for list_g in gallery_list:      
            if list_g[0:7] == str(int(gallery_label[q]))+"_c"+str(int(gallery_cam[q])):
                shutil.copy(main_gallery_dir+"/"+str(int(gallery_label[q]))+"/"+list_g,main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q]))+"/"+list_g)





    else:            
        gallery_list = os.listdir(main_gallery_dir+"/"+str(int(gallery_label[q])).zfill(4))
        
        for list_g in gallery_list:      
            if list_g[0:7] == str(int(gallery_label[q])).zfill(4)+"_c"+str(int(gallery_cam[q])):
                shutil.copy(main_gallery_dir+"/"+str(int(gallery_label[q])).zfill(4)+"/"+list_g,main_dir+"/"+str(int(gallery_cam[q]))+"/"+str(int(gallery_label[q])).zfill(4)+"/"+list_g)






for c in range(8):
    if os.path.exists(main_dir+"/"+str(c+1)+"/"+str(-1)):
        shutil.rmtree(main_dir+"/"+str(c+1)+"/"+str(-1))











