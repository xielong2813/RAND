## RAND

---
### Introduction
This repo is a Pytorch implementation of the paper--RAND:Adaptive Normalization and Denormalization Method for Non-Stationary Time Series Forecasting.

RAND is a plug-and-play normalization and denormalization method, namely Resolution-Adaptive Normalization and Denormalization, it is devised to deal with the distribution shift problem in time series for machine-learning-based forecasting model. It normalizes the input time series to reduce distribution differences between
instances and addaptively denormalizes the output series by modeling variations of slice-level time-varying mean and variance.


### Usage
### 🛠 Prerequisites
#### Environment and dataset setup

Ensure you are using Python 3.9 and install the necessary dependencies by running:

```bash
pip install -r requirements.txt
mkdir datasets
```
### 📊 Prepare Datastes
All the datasets are available at the [Google Driver](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by Autoformer. 
Begin by downloading the required datasets. All datasets are conveniently available at Autoformer. Create a separate folder named ./dataset and neatly organize all the csv files as shown below:
```
dataset
└── electricity.csv
└── ETTh1.csv
└── ETTh2.csv
└── ETTm1.csv
└── ETTm2.csv
└── traffic.csv
└──  weather.csv
```
#### Running

We provide ready-to-use scripts for RAND enhanced backbone models.

```bash
sh run_rand.sh
```

#### 🙏 Acknowledgement
Special thanks to the following repositories for their invaluable code and datasets:

https://github.com/thuml/Autoformer

https://github.com/honeywell21/DLinear

https://github.com/cure-lab/LTSF-Linear

https://github.com/icantnamemyself/SAN

https://github.com/wanghq21/MICN

https://github.com/thuml/Time-Series-Library

https://github.com/MAZiqing/FEDformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/weifantt/Dish-TS

https://github.com/yuqinie98/PatchTST

https://github.com/Thinklab-SJTU/Crossformer

### 📩 Contact
If you have any questions, please contact xiel21@mails.tsinghua.edu.cn or submit an issue.
