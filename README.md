# Person-reID
ISSUM person-reID project
This project is focused on the unsupervised person-reid that is applicable to different domains.
## GAN-based Unsupervised Person-reid
explain the method here


## steps:

### Prepare datasets
Prepare the datasets to the common format,for training and test procedures (taken from st-reid Github)
prepare.py --Market   //  prepare.py --Duke

### train the person re-id model
python3 train_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "./dataset/market_rename/"
python3 train_duke.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "./dataset/DukeMTMC_prepare/"

### extract features via pre-trained model
**source:** market        **target:** duke

python3 test_st_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --test_dir "./dataset/DukeMTMC_prepare/"

**source:** market        **target:** market

python3 test_st_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --test_dir "./dataset/market_rename/"

**source:** duke          **target:** duke

python3 test_st_duke.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_duke_e --test_dir "./dataset/DukeMTMC_prepare/"

**source:** duke           **target:** market

python3 test_st_duke.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_duke_e --test_dir "./dataset/market_rename/"

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
