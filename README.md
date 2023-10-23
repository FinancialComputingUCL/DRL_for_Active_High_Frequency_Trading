# DRL for active HFT

We propose an open-source version of the code for the first end-to-end Deep Reinforcement Learning based framework for active high-frequency trading.

### Authors
- Antonio Briola
- Jeremy Turiel
- Riccardo Marcaccioli
- Alvaro Cauderan
- Tomaso Aste

### Code author 
- Alvaro Cauderan


### Reference paper
[DRL for Active HFT](https://arxiv.org/pdf/2101.07107.pdf)

### Description

We train DRL agents to trade one unit of Intel Corporation stocks by employing the Proximal Policy Optimization algorithm. The training is performed on three contiguous months of high-frequency Limit Order Book data. In order to maximize the signal-to-noise ratio in the training data, we compose the latter by only selecting training samples with the largest price changes. The test is then carried out on the following month of data. Hyperparameters are tuned using the Sequential Model Based Optimization technique. We consider three different state characterizations, which differ in the LOB-based meta-features they include. Agents learn trading strategies able to produce stable positive returns in spite of the highly stochastic and non-stationary environment, which is remarkable itself. Analyzing the agents’ performances on the test data, we argue that the agents are able to create a dynamic representation of the underlying environment highlighting the occasional regularities present in the data and exploiting them to create long-term profitable trading strategies.

### Data description

We use data from the INTEL stock from the [LOBSTER](https://lobsterdata.com) database. Unfortunately, we are not able to share them as they are proprietary data.

Nonetheless, if you are able to acquire it through appropriate means, here is how it should be structured in the code:

    .
    ├── ...
    ├── data      
    │   ├── clean_train_data    # Keep empty
    │   ├── test_data           # Add raw testing csv's
    │   └── train_data          # Add raw training csv's
    └── ...

### Code usage

The code can be run through main.py and the parameters can be set through the argument parser (each parameter is explained in the code).

### How to cite

@article{briola2021deep,\
  title={Deep reinforcement learning for active high frequency trading},\
  author={Antonio Briola and Jeremy Turiel and Riccardo Marcaccioli and Alvaro Cauderan and Tomaso Aste},\
  journal={arXiv preprint arXiv:2101.07107},\
  year={2021}
}

### Contacts

a.briola@ucl.ac.uk \
jeremy.turiel.18@ucl.ac.uk \
riccardo.marcaccioli@ladhyx.polytechnique.fr \
acauderan@ethz.ch \
t.aste@ucl.ac.uk 
