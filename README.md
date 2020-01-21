# Analysis Dictionary Learning Based Classification: Structure for Robustness
This repository contains the demo and MATLAB codes for our IEEE TIP paper: "[Analysis Dictionary Learning Based Classification: Structure for Robustness](https://arxiv.org/pdf/1807.04899.pdf)" by [Wen Tang](https://research.ece.ncsu.edu/vissta/wen-tang/), Ashkan Panahi, Hamid Krim and Liyi Dai. The Algorithm proofs can be found in the [supplementary material](https://www.researchgate.net/publication/334048407_bare_conf_compsocSupppdf).

This is also the journal version of the conference paper "[Structured Analysis Dictionary Learning for Image Classification](https://arxiv.org/abs/1805.00597)", which is published in ICASSP 2018.

### Citation
If you think our projects are useful, please consider citing them:
```
@article{tang2019analysis,
  title={Analysis dictionary learning based classification: Structure for robustness},
  author={Tang, Wen and Panahi, Ashkan and Krim, Hamid and Dai, Liyi},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={12},
  pages={6035--6046},
  year={2019},
  publisher={IEEE}
}

@inproceedings{tang2018structured,
  title={Structured analysis dictionary learning for image classification},
  author={Tang, Wen and Panahi, Ashkan and Krim, Hamid and Dai, Liyi},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2181--2185},
  year={2018},
  organization={IEEE}
}
```

## Introduction
A discriminative structured analysis dictionary is proposed for the classification task. A structure of the union of subspaces (UoS) is integrated into the conventional analysis dictionary learning to enhance the capability of discrimination. A simple classifier is also simultaneously included into the formulated function to ensure a more complete consistent classification. The solution of the algorithm is efficiently obtained by the linearized alternating direction method of multipliers. Moreover, a distributed structured analysis dictionary learning is also presented to address large scale datasets. It can group-(class-) independently train the structured analysis dictionaries by different machines/cores/threads, and therefore avoid a high computational cost. A consensus structured analysis dictionary and a global classifier are jointly learned in the distributed approach to safeguard the discriminative power and the efficiency of classification. Experiments demonstrate that our method achieves a comparable or better performance than the state-of-the-art algorithms in a variety of visual classification tasks. In addition, the training and testing computational complexity are also greatly reduced.
### Structured Analysis Dictionary Learning (SADL)
<img src="https://latex.codecogs.com/svg.latex?\arg\min_{\substack{\Omega,U,Q,W,\\&space;\varepsilon_1,\varepsilon_2}}\frac{1}{2}\|U-\Omega&space;X\|_F^2+\lambda_1\|U\|_1+\frac{\rho_1}{2}\|\varepsilon_1\|^2_2+\frac{\rho_2}{2}\|\varepsilon_2\|^2_2" title="SADL1" />
<img src="https://latex.codecogs.com/svg.latex?+\frac{\delta_1}{2}\|Q\|^2_2+\frac{\delta_2}{2}\|W\|^2_2+\frac{\lambda_2}{2}\|\Omega\|^2_2" title="SADL2" />
<img src="https://latex.codecogs.com/svg.latex?+\gamma_1\|H-QU\|^2_F+\gamma_2\|L-W(QU)\|_F^2" title="SADL3" />
<img src="https://latex.codecogs.com/svg.latex?\emph{s.t.}~H=QU+\varepsilon_1,~Y=W(QU)+\varepsilon_2,~|\omega_i^T\|_2^2=1,~\forall&space;i=1,\dots,r." title="SADL4" />

### Distributed SADL
<img src="https://latex.codecogs.com/svg.latex?\arg\min_{\substack{\Omega_t,U_t,\\Q_t,W_t,\\&space;\Omega,Q,W,\\&space;\varepsilon_{1_t},\varepsilon_{2_t}}}\sum_{t=1}^N(\frac{1}{2}\|U_t-\Omega_t&space;X_t\|_F^2+\lambda_1\|U_t\|_1+\frac{\rho_{1_t}}{2}\|\varepsilon_{1_t}\|^2_2" title="DSADL1" />
<img src="https://latex.codecogs.com/svg.latex?+\frac{\rho_{2_t}}{2}\|\varepsilon_{2_t}\|^2_2+\frac{\xi_{1_t}}{2}\|\Omega-\Omega_t\|^2_2+\frac{\delta_{1_t}}{2}\|Q_t\|^2_2+\frac{\lambda_{2_t}}{2}\|\Omega_t\|^2_2" title="DSADL2" />
<img src="https://latex.codecogs.com/svg.latex?+\frac{\xi_{2_t}}{2}\|Q-Q_t\|^2_2+\frac{\delta_{2_t}}{2}\|W_t\|^2_2+\frac{\xi_{3_t}}{2}\|W-W_t\|^2_2)" title="DSADL3" />
<img src="https://latex.codecogs.com/svg.latex?+\gamma_1\|H-QU\|^2_F+\gamma_2\|L-W(QU)\|_F^2" title="DSADL4" />
<img src="https://latex.codecogs.com/svg.latex?\emph{s.t.}~H_t=Q_tU_t+\varepsilon_{1_t},~Y_t=W_t(Q_tU_t)+\varepsilon_{2_t}," title="DSADL5" />
<img src="https://latex.codecogs.com/svg.latex?\emph{&space;&space;&space;&space;}~\|\omega_{i}^T\|_2^2=1;&space;\forall&space;i=1,\dots,r,~\|\omega_{t_i}^T\|_2^2=1;~\forall&space;i=1,\dots,r,~\forall&space;t=1,\dots,N." title="DSADL6" />

## Contacts
email: wtang6@ncsu.edu
