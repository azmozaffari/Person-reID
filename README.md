# A Simple and Effective Person-reID

This project is focused on the domain adaptive person re-identification in intra-camera supervised setting.











We have proposed a new two-level coarse to fine  UAD approach to improve the accuracy rate of unsupervised re-id matching in different domains

# STN-CAD Person re-id
The algorithm contains two main parts:
1- STN: extracts preliminarily labels for the target domain using the pre-trained source model and adding spatio-temporal features to boost the results.
2- CAD: A teacher student model to learn feature representation for the target domain using the preliminarily labels provided from STN part.
## STN
### Prepare datasets
put downloaded datasets in original_dataset folder as follow: 

```
./dataset
├── modified_dataset
│   ├── Duke
│   │   ├── gallery
│   │   ├── query
|   |   ├── multi_query
│   │   ├── train
│   │   ├── train_all
│   │   └── val
│   └── Market
│       ├── gallery
│       ├── query
|       ├── multi_query
│       ├── train
│       ├── train_all
│       └── val
└── original_dataset
    ├── Duke
    │   ├── bounding_box_test
    │   ├── bounding_box_train
    │   └── query
    └── Market
        ├── bounding_box_test
        ├── bounding_box_train
        ├── gt_bbox
        ├── gt_query
        └── query



```
python3 prepare_dataset.py --dataset_name market --dataset_folder ./dataset  //  prepare.py --duke

### Prepare pre-trained model
we use PCB model for training and save the trained models in ./model folder

python3 train.py --PCB --gpu_ids 0 --model_name ft_ResNet50 --erasing_p 0.5 --train_all --train_dir "./dataset/modified_dataset/" --source market

train without PCB

python3 train.py  --gpu_ids 0 --model_name ft_ResNet50 --erasing_p 0.5 --train_all --train_dir "./dataset/modified_dataset/" --source market


### Extract features using pre-trained model

query_type is "query" or "multi-query" depends on the data (tracklet or single query)

python3 extract_features.py --PCB --gpu_ids 0  --source market --target duke --query_type multi_query

it saves two files in ./rep folder one contains features of all single imgs, the other takes average of feature vector and frame numbers for each tracklet and save that with suffix _s 


### Extract cosine features
for two types query and multi_query it extracts cosine features
--S for tracklets is saved as mean value

python3 extract_cosine_scores.py --PCB --gpu_ids 0 --source market --target market --query_type query  --S

### Main
run the code

python3 python3 main.py --PCB --gpu_ids 0 --source duke --target market --query_type query



