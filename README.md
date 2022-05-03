# CPSC 322 Final Project
## Nelly Alger & Maya Fleming

This project performs classification of stroke data with 4 different classifiers, predicting whether a patient is likely to have a stroke or not based on parameters like gender, age, various diseases, and smoking status.

## Project Structure
* `Technical_Report.ipynb` describes our approach and findings, with interleaved code cells running our data cleaning, exploratory data analysis, and classification methods. 
    * **To run the project and view results: run all cells in `Technical_Report.ipynb`**
* The folders `mysklearn` and `myEDA`: contain all classification and EDA source code
* `----_work.py` files: call the functions from mysklearn and myEDA
    * These work files are essentially performing the EDA and classification but are stored outside of the report in order to reduce clutter
* The folder `input_data`: contains the original dataset csv file: `stroke_data.csv`, as well as various cleaned versions of the dataset
* `project_app.py` contains the flask app of our best classifier 

### Other Files
* `test_myclassifiers`: demonstrates our test driven development, and houses the tests for our various classifiers. 
    * The tests for the Random Forest Classifier were implemented for this project specifically, while the others existed from previous projects
* `Project_Proposal.ipynb`: original project proposal 
* `data_notes.txt`: contains additional information about each attribute and all possible values, used for discretization and data cleaning


All code is written by Nelly and Maya.