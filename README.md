# ADOD

hkust-CSIT5210-Group3-project

github repo link: https://github.com/xdl2003/ADOD

This project is developed by the CSIT5210-GROUP3 team, focusing on the implementation of the ADOD (Anomaly Detection based on Density) algorithm.


## Project Overview
This project implements the density-based anomaly detection algorithm ADOD. It includes:
- One synthetic dataset
- Two real-world datasets
- Performance evaluation of ADOD and baseline algorithms using ROC-AUC and p@n metrics across different datasets


## Environment Requirements
- Python 3.11.2

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
Execute the main function in ADOD/code/main.py.
## Key Code Files
- ADOD/code/main.py: Input/output processing and implementation of the KNN-based baseline algorithm
- ADOD/code/adod.py: Implementation of the ADOD algorithm
- ADOD/code/genxxx.py: Input data processing