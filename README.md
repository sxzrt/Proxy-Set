# Ranking Models in Unlabeled New Environments


## Prerequisites
This code uses the following libraries

- Python 3.7
- NumPy
- PyTorch 1.7.0 + torchivision 0.8.1
- Sklearn
- Scipy 1.2.1

the environment can be created by using "proxy_set.yml" :
```shell script
conda env create -f proxy_set.yml 
```

## Data Preparation
The folder of each dataset (take Market-1501 as an example) in the data pool should look like this:
```
Market-1501
├── bounding_box_train/  # the traning set is only necessary for target dataset
│   └── ...
├── bounding_box_test/ 
│   └── ...
└── query/
    └── ...
```

## Run the Code
### searching data 
```shell script
python dataset_selection.py --weight 0.6 -- result_dir 'sample_data/'
``` 

Searched data will be saved in "result_dir". Other parameters, such as the number of clusters, can be set in dataset_selection.py.

## Citation
Please cite this paper if it helps your research:
```bibtex
@inproceedings{sun2021,
  title={Ranking Models in Unlabeled New Environments},
  author={Sun, Xiaoxiao and Hou, Yunzhong and Deng, Weijian and Li, Hongdong and Zheng, Liang},
  booktitle={IEEE Conference on International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

