# CPSC 322 Final Project: 
## Nelly Alger & Maya Fleming

This project performs classification of stroke data, predicting whether a patient is likely to have a stroke or not based on parameters like gender, age, various diseases, and smoking status.

## Project Structure
* `Technical_Report.ipynb` describes our approach and findings, with interleaved code cells running our data cleaning, exploratory data analysis, and classification methods. 
* The folder`mysklearn`: contains all classification source code is located in , and called in the various `----_work.py` files
    * These files are imported and called with functions in the Technical report in order to reduce clutter in the report
    * **To run the project and view results: run all cells in `Technical_Report.ipynb`**
* `Project_Proposal.ipynb`: original project proposal 
* The folder `input_data`: contains the original dataset csv file: `stroke_data.csv`, as well as various cleaned versions of the dataset
* `test_myclassifiers`: demonstrates our test driven development, and houses the tests for our various classifiers. 
    * The tests for the Random Forest Classifier were implemented for this project specifically, while the others existed from previous projects


All code is written by Nelly and Maya.