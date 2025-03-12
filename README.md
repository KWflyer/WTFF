# Multi-Domain Time-Frequency Fusion Feature Contrastive Learning for Machinery Fault Diagnosis-- Wide Kernel Time-Frequency Fusion （WTFF）

The source code is for the following paper which will be available soon!

Yang Wei，Kai Wang. Multi-Domain Time-Frequency Fusion Feature Contrastive Learning for Machinery Fault Diagnosis[J]. IEEE Signal Processing Letters, 2025.

If you find this code is useful and helpful to your work, please cite our paper in your research work. Thanks.

If there are any questions about source code, please do not hesitate to contact Yang Wei(weiyangkabu@stu.scu.edu.cn) and me (kai.wang@scu.edu.cn). Special thanks to Rui Luo for helping to arrange and re-implement the code.

## How to use the code
### Running environment:
The proposed methods are implemented in Python 3.9.7 with PyTorch 1.12.0 framework on a desktop computer equipped with an Intel i9 12900k CPU and an NVIDIA RTX 3090 GPU.

### Dataset used in this paper:
1.	[PU](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download)
2.	[CWRU](https://pan.baidu.com/s/1Z1pznf-snyXB5jBzlXbUEw?pwd=4fpx#list/path=%2F&parentPath=%2Fsharelink667481542-561356365509374)

### How to reproduce the experimental results of surface defect classification.
1.	find the `cal_101_resnet.py` file in classification folder:
2.	configurable arguments:
    ``` train.py [--link_place LINK_PLACE] [--re_co RE_CO] [--backbone BACKBONE] ```
3.	train your backbone
4.	evaluating results will automatically show after training.
