# AI_MPM
Code availability repository

this is a Python program for AI MPM project; here is 3 brunches for each optimizer. see GS MLP/ SSA MLP/GWO MLP for code availability

for running program make sure that all requirement libraries are installed

As pre process, we need 2 .csv files as train and main databases. in train database, each point has cordinate, features and a binary "label" columns. 
main database has same structures, but without label column. 

before running code, add names of features are in modeling in "features" box in code. 
example:
features = ['g1','g2','g3','g4','r1','r2','r3','r4','r5','r6','a1','a2','a3','a4','f1','f2']

labels are 0 for non-deposits and 1 for deposits
code will read numbers in "label" column as training labels


now Run the code

at first, code will ask for training dataset file path
type the file path. 
example: d:/train.csv

then, code ask for main dataset
type the file path. 

example: d:/main.csv

now, code will run, at first code will optimize MLP parameters using train dataset to reach optimum, then code will process main database and save new .csv file including all points with range of labels [0-1]
