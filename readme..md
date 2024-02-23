Q1: gather, clean and display data. input is limited to a single samle of max price for day
Q2: train classic AE_MNIST on the data. use cross-training and test on regular batches
Q3: icorporate prediction
Q4: adjust into multi-step


#Q1:
- talk about how the reconstruction is almost accurate but not exactly (meaning a good prediction and also not learning the ID function)
- talk about how each parameter in the parser affected results


# Q2:
- show a run of the V1 MNIST preformance with explanations of the exploration of reconstruction/classification loss ratio, layers of MNIST, etc. 
- show a run of the other versions with different places of classifications.


at the end:
- make all dry_runs=True and organise Tests.py
- remove prints
- change default parameters for all models