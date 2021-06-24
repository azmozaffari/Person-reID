# Person-reID
ISSUM person-reID project
This project is focused on the unsupervised domain adaptive person re-identification.
We have proposed a new two-part UAD approach to improve the accuracy rate of unsupervised re-id matching in different domains

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
### extract features using pre-trained model
python3 test.py --PCB --gpu_ids 0  --source duke --target market --query_type multi_query  

it saves the file in ./rep folder


###  Evaluate  


### extract cosine features
python3 cosine_feature_extraction.py --source_domain duke --target_domain duke

### time distribution extraction
extract the simple distribution of occurance between two cameras in the specific target domain
source domain here mentioned that we are using the features extracted from thetarget domain by apllying the pretraoined model of the source domain

python3 time_distribution_function.py  --source_domain market --target_domain duke

### draw different histograms
in draw hist.py file there are two functions for drawing the time difference histogram for all combinations of the camera and drawing features cosine similarity for different domains.
python3 draw_hist.py --source_domain market --target_domain duke


### extrcat hard negative samples
The samples that have cosine similarity greater than the threshold will be extracted and saved in the hard_negative_samples dir 
pyhton3 hard_negative_samples_mining.py --gpu_ids 0 --source_domain market --target_domain duke  --threshold 0.7

### train GAN for negative samples
### extract time lift distribution
### evaluate the results
### re-rank the results
