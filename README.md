#Code Availability
. 
.. 
This repository contains a Python implementation for the AI-MPM project.
The project includes three separate branches, each corresponding to a different optimizer: GS-MLP, SSA-MLP, and GWO-MLP.
Please switch to the desired branch to access the related code.


Requirements:
Before running the program, make sure that all required Python libraries are installed on your system.


Input Data Format:
As a preprocessing step, two CSV files are required.
The first file is the training dataset. Each record represents a data point and must include coordinate columns, feature columns, and a binary label column.
Label values are defined as follows:
0 represents non-deposit areas and 1 represents deposit areas.
The second file is the main dataset. It has the same structure as the training dataset but does not contain the label column.


Feature Definition:
Before running the program, the names of the features used in the model must be defined in the features section of the code.
These feature names should exactly match the column names in the CSV files.
For example, the features may include geochemical, geophysical, alteration, and fault-related variables.


How to Run:
After starting the program, the user is prompted to enter the file path of the training dataset.
After that, the file path of the main dataset is requested.
Processing Workflow
First, the program optimizes the MLP parameters using the training dataset to reach the optimal configuration.
Then, the optimized model is applied to the main dataset.
The final output is a new CSV file containing all data points along with predicted label values ranging from 0 to 1.


Example Files:
For more information about the input data structure and formatting, refer to the example files named trn.csv and tst.csv.
