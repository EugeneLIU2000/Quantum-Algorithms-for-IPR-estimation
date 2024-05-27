# Quantum-Algorithms-for-IPR-estimation

These notes are code implementations for the propsed algorithm in the article https://arxiv.org/pdf/2405.03338, the data and figures in the article are included in this GitHub repository.

### Overview of Repository Structure
This repository contains a series of scripts configured to test the performance of Quantum Neural Networks across various datasets and training conditions. Each script is prefixed with a code indicating the target dataset:

* BC: Scripts for running classifiers on the Breast Cancer dataset.
* FMNIST: Scripts for running classifiers on the Fashion MNIST dataset.
* MNIST: Scripts for running classifiers on the MNIST dataset.

#### Script Description
* `PXP`: Contains code for simulation and data for Fig.5. The examplary code requires the installation of Python package `quspin` (https://quspin.github.io/QuSpin/).
  
* `OAT`: Includes simulation and data for Fig.4 and experimental data for Fig.7. To obtain the experimental results, one could install ibm-qiskit to run the provided file `OAT-IBM_device` with their own ibm cloud service and keys.

* `AKLT`: Simulation and data for Fig.6.
