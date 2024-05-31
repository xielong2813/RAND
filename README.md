## RAND

---
### Introduction
This repo is a Pytorch implementation of the paper--RAND:Adaptive Normalization and Denormalization Method for Non-Stationary Time Series Forecasting.

RAND is a plug-and-play normalization and denormalization method, namely Resolution-Adaptive Normalization and Denormalization, it is devised to deal with the distribution shift problem in time series for machine-learning-based forecasting model. It normalizes the input time series to reduce distribution differences between
instances and addaptively denormalizes the output series by modeling variations of slice-level time-varying mean and variance.


### Usage

#### Environment and dataset setup

```bash
pip install -r requirements.txt
mkdir datasets
```

All the datasets are available at the [Google Driver](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by Autoformer. Thanks to their work!

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
