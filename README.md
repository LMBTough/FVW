<div align="center">

# FVW: Identifying and Leveraging Valuable Weights in Deep Neural Networks for Efficient Model Pruning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue:CIKM 2023](https://img.shields.io/badge/Venue-CIKM%202023-007CFF)](https://uobevents.eventsair.com/cikm2023/)

</div>


## Abstract
This repository houses the official implementation of the "FVW: Finding Valuable Weight on Deep Neural Network for Model Pruning" methodology, a novel approach aimed at enhancing the computational efficiency of deep neural networks. By identifying and leveraging the most valuable weights within a network, our method significantly reduces the model's complexity without compromising its predictive performance. This technique is particularly beneficial for deployment in resource-constrained environments.

## Prerequisites
Before proceeding, ensure the model weights are prepared:
- Download the pre-pruning model weights from [this link](https://drive.google.com/file/d/1UXGUXzU83i69bJRyioAuuEUDQ0STAsgq/view?usp=share_link).
- Extract the weights archive into the `masks/weights` directory, ensuring the model is ready for pruning and further experimentation.

## Installation
To set up the environment for running the FVW implementation, execute the following command to install the necessary Python dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

### Dependencies
- numpy==1.23.3
- torch==1.12.1+cu113
- torchvision==0.13.1+cu113

## Methodology
The FVW approach integrates adversarial attack strategies with a novel attribution algorithm to dissect and evaluate the contribution of individual weights within the network. This granular analysis facilitates a more informed and effective pruning strategy, preserving the integrity and performance of the pruned model.

## Usage
To employ the FVW methodology for model pruning:
1. Prepare your environment as per the Installation section.
2. Load your model and the corresponding pre-pruning weights.
3. Follow the step-by-step guide in `main.ipynb` to apply the FVW pruning process to your model.

## Citing FVW
If you utilize this implementation or the FVW methodology in your research, please cite the following paper:

```
@inproceedings{zhu2023fvw,
  title={FVW: Finding Valuable Weight on Deep Neural Network for Model Pruning},
  author={Zhu, Zhiyu and Chen, Huaming and Jin, Zhibo and Wang, Xinyi and Zhang, Jiayu and Xue, Minhui and Lu, Qinghua and Shen, Jun and Choo, Kim-Kwang Raymond},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3657--3666},
  year={2023}
}
```

## Acknowledgments
We extend our gratitude to the contributors and researchers whose insights and efforts have been instrumental in the development of the FVW methodology.

For further information or inquiries, please refer to the corresponding author(s) of the FVW paper or initiate a discussion in this repository's Issues section.
