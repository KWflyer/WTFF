# Multi-Domain Time-Frequency Fusion Feature Contrastive Learning for Machinery Fault Diagnosis-- Wide Kernel Time-Frequency Fusion （WTFF）

The source code is for the following paper :

Wei Y, Wang K. Multi-Domain Time-Frequency Fusion Feature Contrastive Learning for Machinery Fault Diagnosis[J]. IEEE Signal Processing Letters, 2025 (99): 1-5.（https://ieeexplore.ieee.org/document/10910177）

If you find this code is useful and helpful to your work, please cite our paper in your research work. Thanks.

If there are any questions about source code, please do not hesitate to contact Yang Wei(weiyangkabu@stu.scu.edu.cn) and me (kai.wang@scu.edu.cn). Special thanks to Rui Luo for helping to arrange and re-implement the code.

## How to use the code
### Running environment:
The proposed methods are implemented in Python 3.9.7 with PyTorch 1.12.0 framework on a desktop computer equipped with an Intel i9 12900k CPU and an NVIDIA RTX 3090 GPU.

### Dataset used in this paper:
1.	[PU](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download)
2.	[JNU](https://github.com/ClarkGableWang/JNU-Bearing-Dataset)

### How to reproduce the experimental results of  Machinery Fault Diagnosis.
1.  Download the required datasets:  According to the provided dataset addresses, download the required datasets for the experiment and store them in the respective subfolders under the `dataset` folder.

2.  Configure Environment：  Find the `requirements.txt` file in WTFF folder and configure Environment：``` pip install -r requirements.txt ```
3.	 Configurable arguments:  Find the `run_main.py` file in WTFF folder，According to the experiment you are going to conduct, set the parameters in the `run_main.py` file.
   For  example：
5.	Train your backbone and 
6.	train your backbone
7.	evaluating results will automatically show after training.
