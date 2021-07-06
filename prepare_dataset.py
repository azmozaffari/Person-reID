import os
from shutil import copyfile
import argparse
from os import listdir
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='Preparing Datasets with and without tracklets for query')
parser.add_argument('--dataset_name', default='duke', type=str,help='duke, market, msmt')
parser.add_argument('--dataset_folder', default='./dataset', type=str, help='dataset folder address')

opt = parser.parse_args()

dataset_name = opt.dataset_name
dataset_folder = opt.dataset_folder


original_dataset_folder = dataset_folder+"/"+"original_dataset"+"/"+dataset_name
modified_dataset_folder = dataset_folder+"/"+"modified_dataset"+"/"+dataset_name




if dataset_name=="market":
    save_path = "./dataset/Market1501_prepare/"
    download_path = dataset_folder+"/"+"original_dataset"+"/"+dataset_name

else:
    save_path = modified_dataset_folder
    download_path = dataset_folder+"/"+"original_dataset"+"/"+dataset_name




if os.path.isdir(modified_dataset_folder):
    shutil.rmtree(modified_dataset_folder)
if os.path.isdir("./dataset/Market1501_prepare/"):    
    shutil.rmtree("./dataset/Market1501_prepare/")   


if not os.path.isdir(dataset_folder+"/"+"modified_dataset"):
    os.mkdir(dataset_folder+"/"+"modified_dataset")
if not os.path.isdir(dataset_folder+"/"+"modified_dataset"+"/"+dataset_name):
    os.mkdir(dataset_folder+"/"+"modified_dataset"+"/"+dataset_name)




if not os.path.exists(save_path):
    os.makedirs(save_path)
# -----------------------------------------
# query
query_path = download_path + '/query'
query_save_path = save_path + '/query'
if not os.path.exists(query_save_path):
    os.makedirs(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# -----------------------------------------
# gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = save_path + '/gallery'
if not os.path.exists(gallery_save_path):
    os.makedirs(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_all
train_path = download_path + '/bounding_box_train'
train_save_path = save_path + '/train_all'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_val
train_path = download_path + '/bounding_box_train'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)
    os.makedirs(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)


# ================================================================================================
# market1501_rename
# ================================================================================================

def parse_frame(imgname, dict_cam_seq_max={}):
    dict_cam_seq_max = {
        11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
        21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
        31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
        41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
        51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
        61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
    fid = imgname.strip().split("_")[0]
    cam = int(imgname.strip().split("_")[1][1])
    seq = int(imgname.strip().split("_")[1][3])
    frame = int(imgname.strip().split("_")[2])
    count = imgname.strip().split("_")[-1]
    # print(id)
    # print(cam)  # 1
    # print(seq)  # 2
    # print(frame)
    re = 0
    for i in range(1, seq):
        re = re + dict_cam_seq_max[int(str(cam) + str(i))]
    re = re + frame
    new_name = str(fid) + "_c" + str(cam) + "_f" + '{:0>7}'.format(str(re)) + "_" + count
    # print(new_name)
    return new_name


def gen_train_all_rename():
    path = "./dataset/Market1501_prepare/train_all/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists(modified_dataset_folder+"/train_all/" + fname):
            os.makedirs(modified_dataset_folder+"/train_all/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = modified_dataset_folder+"/train_all/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_train_rename():
    path = "./dataset/Market1501_prepare/train/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists(modified_dataset_folder+"/train/" + fname):
            os.makedirs(modified_dataset_folder+"/train/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = modified_dataset_folder+"/train/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_val_rename():
    path = "./dataset/Market1501_prepare/val/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists(modified_dataset_folder+"/val/" + fname):
            os.makedirs(modified_dataset_folder+"/val/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = modified_dataset_folder+"/val/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_query_rename():
    path = "./dataset/Market1501_prepare/query/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists(modified_dataset_folder+"/query/" + fname):
            os.makedirs(modified_dataset_folder+"/query/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = modified_dataset_folder+"/query/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_gallery_rename():
    path = "./dataset/Market1501_prepare/gallery/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists(modified_dataset_folder+"/gallery/" + fname):
            os.makedirs(modified_dataset_folder+"/gallery/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = modified_dataset_folder+"/gallery/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id

def gen_multi_query_rename():
    path = "./dataset/Market1501_prepare/multi_query/"
    
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists(modified_dataset_folder+"/multi_query/" + fname):
            os.makedirs(modified_dataset_folder+"/multi_query/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = modified_dataset_folder+"/multi_query/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个i

def gen_multi_query():
#     multi_query_folder = modified_dataset_folder+"/"+"multi_query"
    if dataset_name == 'market':
        multi_query_folder =  "./dataset/Market1501_prepare"+"/"+"multi_query"
    else:
        multi_query_folder = modified_dataset_folder+"/"+"multi_query"

    if os.path.isdir(multi_query_folder):
        shutil.rmtree(multi_query_folder)
    if not os.path.isdir(multi_query_folder):
        os.mkdir(multi_query_folder)
    file_names_query = listdir(original_dataset_folder+"/query")
    file_names_gallery = listdir(original_dataset_folder+"/"+"bounding_box_test")
    for file in file_names_query:
        if file !="Thumbs.db":
            src_path = original_dataset_folder+"/"+"query"+"/"+file
            g_src_path = original_dataset_folder+"/"+"bounding_box_test"
            dst_path = multi_query_folder

            ID = file[0:4]
            cam = file[6]

            if not os.path.isdir(multi_query_folder+"/"+ID):
                os.mkdir(multi_query_folder+"/"+ID)
            copyfile(src_path, dst_path + '/' +ID+"/" +file)

            for g_file in file_names_gallery:
                if file[0:7] in g_file:
                    copyfile(g_src_path+"/"+g_file, dst_path + '/' +ID+"/" +g_file)






gen_multi_query()

if dataset_name=="market":
    gen_train_all_rename()
    gen_train_rename()
    gen_val_rename()
    gen_query_rename()
    gen_gallery_rename()
    gen_multi_query_rename()
    shutil.rmtree("./dataset/Market1501_prepare/")   
print("Done!")



