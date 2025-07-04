# BINTS: Bi-Modal Learning for Networked Time Series

This is the implementation of BINTS published in KDD 2025 [[paper]](https://github.com/kaist-dmlab/BINTS)

## 1. Overview
Understanding human mobility patterns is a complex challenge that requires modeling both node features and node interactions within graph topologies across time. While previous methods have focused on either node features or interactions, the synergistic integration of these two modalities has proven difficult to achieve. In this paper, we propose ***BINTS*** (**BI**-modal learning for **N**etworked **T**ime **S**eries), a pioneering bi-modal learning framework that employs soft contrastive learning along the temporal axis. ***BINTS*** captures modality similarities and temporal patterns by simultaneously learning from evolving node features and interactions, solving the limitations of single-modality approaches. To evaluate our method, we curate comprehensive multi-modal human mobility datasets spanning diverse locations and times. Our experimental results demonstrate that BINTS significantly outperforms existing forecasting models by capturing synergies across different data modalities. Overall, we establish BINTS as a powerful technique for holistically understanding and forecasting complex mobility dynamics.

## 2. Proposed Benchmark Datasets
Due to the big dataset size, we released it on the anonymous drive
- For research [drive](https://drive.google.com/file/d/1x07YXYeXXMBf7TzVCkLm4_Bfg3c49hEP/view?usp=sharing)
- For benchmark [drive](https://drive.google.com/file/d/1VcFUzvVFZutIKI2OD8bbwEoQX3uGLapV/view?usp=sharing)

| **Domain**       | **Dataset** | **#Node** | **Target Dim.** | **Total Period**             | **Train Days** | **Test Days** | **Time Interval** |
|------------------|-------------|------------|-----------------|------------------------------|----------------|---------------|-------------------|
| **Transportation** | Daegu       | 85         | 7395          | 2022.01.01 ~ 2022.12.31      | 274            | 91            | 1 hour            |
| **Transportation** | Busan       | 103        | 10815           | 2022.01.01 ~ 2022.12.31      | 274            | 91            | 1 hour            |
| **Transportation** | Seoul       | 233        | 54755           | 2022.01.01 ~ 2022.12.31      | 274            | 91            | 1 hour            |
| **Transportation** | NYC Taxi    | 10         | 120             | 2014.01.01 ~ 2023.12.31      | 2739           | 913           | 1 hour            |
| **Epidemic**     | COVID       | 16         | 272             | 2020.01.20 ~ 2023.08.31      | 990            | 330           | 1 day             |
| **Epidemic**     | NYC COVID   | 5          | 30              | 2020.03.01 ~ 2023.12.31      | 1051           | 350           | 1 day             |


## 3. Requirements and Installations
- [Node.js](https://nodejs.org/en/download/): 16.13.2+
- [Anaconda 4](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11.5 (Recommend Anaconda)
- Ubuntu 18.04.6 LTS
- pytorch >= 2.1.2

## 4. Configuration
BINTS was implemented in **Python 3.11.5.**
- Edit main.py file to set experiment parameters (dataset, seq_length, gpu_id(e.g. 0,1,2,3,4,5), etc.)
```bash
python3 main.py
```

## 5. How to run
- Parameter options
```bash
--dataset: the name of dataset (string)
--seq_day: the size of a lookback window (integer)
--pred_day: the size of a prediction window (integer)
```

- At current directory which has all source codes, run main.py with parameters as follows.
```bash
- dataset: {busan, daegu, seoul, covid, nyc, nyc_covid}
- loop: {0, 1, 2, 3, 4}                       # seed for 5-fold evaluation.
- gpu_id: an integer gpu id.

e.g.) python3 main.py --gpu_id 1 --multi_gpu 1 --batch_size 8 --dataset nyc --seq_day 4 --pred_day 7 --khop 5
```
