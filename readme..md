Q1: gather, clean and display data. input is limited to a single samle of max price for day
Q2: train classic AE_MNIST on the data. use cross-training and test on regular batches
Q3: icorporate prediction
Q4: adjust into multi-step

Q2:
###OPTION1####:
try to move to data squence composed of a single date.
####option2####:
try to just feed the model with the table as it is


Q3:
predicting a single stock_value will demand getting it's index in the alphabetical (permanant) order inside the major stocks. then it will be inserted to an array of dummy values for the rest of the stocks and demand prediction.
penalty will be handled by MSE w.r.t the output[stock_index] ONLY



at the end:
- make all dry_runs=True and organise Tests.py
- remove prints
- change default parameters for all models



if i have 1 file called X with path ./A/X and a python script in ./B and i would like to import X to the script - how would i implement it so that it will be valid for all operating systems? 



P3