## RAND

---
### Introduction
This repo is a Pytorch implementation of the paper--RAND:Adaptive Normalization and Denormalization Method for Non-Stationary Time Series Forecasting.


### Usage

#### Environment and dataset setup

```bash
pip install -r requirements.txt
mkdir datasets
```

All the datasets are available at the [Google Driver](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by Autoformer. Thanks to their efforts!

#### Running

We provide ready-to-use scripts for RAND enhanced backbone models.

```bash
sh run_rand.sh
```

#### 🙏 Acknowledgement
Special thanks to the following repositories for their invaluable code and datasets:

https://github.com/thuml/Autoformer
https://github.com/yuqinie98/PatchTST
https://github.com/Thinklab-SJTU/Crossformer
https://github.com/wanghq21/MICN
https://github.com/thuml/Time-Series-Library
https://github.com/MAZiqing/FEDformer
https://github.com/zhouhaoyi/Informer2020
