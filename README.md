# TeleCom-Customer-Churn-ML-prediction
Kaggle Dataset ML with deployment

Dataset from Kaggle: https://www.kaggle.com/datasets/datazng/telecom-company-churn-rate-call-center-data

History:
- Project was done during attatchment at Oak Consulting Singapore (A Beyond Limits Company) from Jun 23 - Jul 23

References:
- https://www.youtube.com/@DigitalSreeni
- https://www.youtube.com/watch?v=bluclMxiUkA&t=918s

Zip folder items:
- data pre-processing folder containing pre-processing code and files
- python files for training ML models
- TelCo-Cust-Churn-Pred-ML-deployement contains everything else

Deployment:
- app run through Python Flask and jinja
- requires pre defined folders

5 Models used and pre-trained on Kaggle dataset: 
- SVM
- Random Forest
- Logistic Regression
- K Nearest Neighbors
- Tensorflow Neural Networks

Deployment:
- 3 main webpages
- displays html table of results
- file data input limited to CSV format
- file results output limited to Excel format

Instructions to run:
- download zip file
- ensure pkl files are in 'models' folder
- ensure updated html files are in the 'templates' folders
- ensure 2 folders (uploads/results ) for user input/output storage
- run app_upload.py pythn script
- run on localhost
- ML test data.csv is the test file to upload and use for prediction   
