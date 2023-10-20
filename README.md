# TeleCom-Customer-Churn-ML-prediction
Kaggle Dataset ML with deployment

Dataset from Kaggle: https://www.kaggle.com/datasets/datazng/telecom-company-churn-rate-call-center-data

History:
- Project was done during attachment at Oak Consulting Singapore (A Beyond Limits Company) from 23 Jun - 28 Jul

References:
- https://www.youtube.com/@DigitalSreeni
- https://www.youtube.com/watch?v=bluclMxiUkA&t=918s

Zip folder items:
- documentations: folder contains useful documentations and instructions for the app
- ML_scripts (REQUIRED): folder contains python scripts and methods essential for the backend
- models (REQUIRED): folder contains the 5 pretrained models, pkl files
- new_models (REQUIRED): folder contains the 5 newly user trained models, pkl files
- results (REQUIRED): folder contains the results of the evaluation of user trained models and prediction results, excel files
- templates (REQUIRED): folder contains the various html files to be rendered through the app.py python flask application
- uploads (REQUIRED): folder contains the csv file uploads from the user
- app.py (REQUIRED): file is the python flask application to run the application
- .git / .vscode / logs : vscode/github folders
- sample data files: folder contains the sample train and predict Tele Communication Customer Churn data from Kaggle, train should be used to train the user trained models and predict is the original dataset from Kaggle used as prediction data 

Deployment:
- app run through Python Flask and jinja
- requires pre defined folders (folders tagged as REQUIRED above)

5 Models used and pre-trained on Kaggle dataset: 
- SVM
- Random Forest
- Logistic Regression
- K Nearest Neighbors
- Tensorflow Neural Networks

Deployment:
- 3 main webpages (Home, Test, Predict)
- displays html table of results
- file data input limited to CSV format
- file results output limited to Excel format
- detailed flow from each html webpage to the next via buttons can be found in the 'documentations' folder named as 'Web_App_Flow.pdf' 

Instructions to run:
- download zip file
- ensure the following 6 folders exist:
    - the app is configured around these folders so it is critical the following folders are named exactly as mentioned and contains the neccessary files
    1. models (contains the 5 different pkl files for each model)
    2. templates (contains the 11 different html files)
    3. ML_scripts (contains 5 model '.py' files, 1 '__init__.py' file)
    4. results
    5. uploads
    6. new_models
- run the app.py script and run the application on the localhost port: 7878
