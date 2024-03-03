
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
- make all dry_runs=False and organise Tests.py
- remove prints
- change default parameters for all models



for github submission:
 - reorganise data into neat folders
 - normalise KFold on reconstruction
 - rewrite all classes and workflows (generating data -> training and testing -> display) in an organised fashion. remove unnecessary codes.

 