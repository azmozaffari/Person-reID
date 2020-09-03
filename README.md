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
source: market  target: duke
python3 test_st_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --test_dir "./dataset/DukeMTMC_prepare/"

source: market target: market
python3 test_st_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --test_dir "./dataset/market_rename/"

source: duke target: duke
python3 test_st_duke.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_duke_e --test_dir "./dataset/DukeMTMC_prepare/"

source:duke target: market
python3 test_st_duke.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_duke_e --test_dir "./dataset/market_prepare/"




### extrcat hard negative samples
### train GAN for negative samples
### extract time lift distribution
### evaluate the results
### re-rank the results
