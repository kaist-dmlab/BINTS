# BINTS

# Bi-Modal Learning for Networked Time Series

## 1. Overview
Understanding human mobility patterns is a complex challenge that requires modeling both node features and node interactions within graph topologies across time. While previous methods have focused on either node features or interactions, the synergistic integration of these two modalities has proven difficult to achieve. In this paper, we propose BINTS (BI-modal learning for Networked Time Series), a pioneering bi-modal learning framework that employs soft contrastive learning along the temporal axis. ***BINTS*** captures modality similarities and temporal patterns by simultaneously learning from evolving node features and interactions, solving the limitations of single-modality approaches. To evaluate our method, we curate comprehensive multi-modal human mobility datasets spanning diverse locations and times. Our experimental results demonstrate that BINTS significantly outperforms existing forecasting models by capturing synergies across different data modalities. Overall, we establish BINTS as a powerful technique for holistically understanding and forecasting complex mobility dynamics. For reproducibility, the source code of our framework is available at anonymous github.


## 2. Public Datasets
| Name          | # Applications    | # Train  | # Test    | Entity×Dimension | # Point Anomaly (Ratio)      | # Pattern Anomaly (Ratio)| Source           |
| :------------:| :----------------:| :------: | :-------: |:----------------:| :---------------------------:| :-----------------------: |:----------------:|
| TODS(Point)   | Synthetic         | 20,000   |  5,000    |  2 × 1           | 250 (100%) | 0 (0%)          |[link](https://github.com/datamllab/tods/tree/benchmark)|
| TODS(Pattern) | Synthetic         | 20,000   |  5,000    |  3 × 1           | 0 (0%)                       | 250 (100%) |[link](https://github.com/datamllab/tods/tree/benchmark)|
| ASD           | Server Monitoring | 8,527   |  4,320     |  12 × 19         | 0 (0%)                       | 199 (100%) |[link](https://github.com/zhhlee/InterFusion) |
| ECG           | Medical Checkup   | 6,995   |  2,851     |  9 × 2           | 0 (0%)    | 208 (100%)       |[link](https://www.cs.ucr.edu/~eamonn/discords/)|
| PSM           | Server Monitoring | 132,481   |  87,841  |  1 × 25          | 16 (0.07%)| 24,365 (99.93%)  |[link](https://github.com/eBay/RANSynCoders/tree/main)|
| Company A     | Server Monitoring | 21,600   |  13,302   |  3 × 8           | 10 (8.53%)                   | 104 (91.47%)            |Private           |

## 3. Requirements and Installations
- [Node.js](https://nodejs.org/en/download/): 16.13.2+
- [Anaconda 4](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8.12 (Recommend Anaconda)
- Ubuntu 16.04.3 LTS
- pytorch >= 1.13.1

## 4. Configuration
Dual-TF was implemented in **Python 3.8.12.**
- Edit main.py and main_freq.py files to set experiment parameters (dataset, seq_length, gpu_id(e.g. 0,1,2,3,4,5), etc.)
```
python3 main.py
python3 main_freq.py
```

## 5. How to run
- Parameter options
```
--dataset: the name of dataset (string)
--seq_length: the size of a window (integer)
--step: the size of a slide (integer)

--input_c: the datasets' dimension (integer) 
--output_c: the datasets' dimension (integer) 
--form: the data form for TODS dataset (string, e.g. point, context, shaplet, seasonal, trend)
--data_num: the number of dataset. (integer)
```

- At current directory which has all source codes, run main.py with parameters as follows.
```
- dataset: {TODS, ASD, ECG, PSM, CompanyA}
- seed: {0, 1, 2}                       # seed for 3-fold cross validation.
- gpu_id: an integer for gpu id.
- seq_length: an integer for outer window length
- nest_length: an integer for inner window length
e.g.) python3 main.py --gpu_id 6 --form seasonal --seq_length 50 --dataset NeurIPSTS --batch_size 4
```
## 6. Licence
Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection © 2023 by Youngeun Nam is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit [(http://creativecommons.org/licenses/by-nc-sa/4.0/)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
