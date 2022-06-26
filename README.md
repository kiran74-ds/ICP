## End to End Machine Learning

The idea behind this project is to develop a Machine Learning model which can be deployed in the production. As part of this I have built a binary classification model on ICP-conform data set .Preliminary Analaysis on data can be found in the Jupyter notebook which resides in research folder.
These are the following items which I have implemented in the code.
+ Create Virtual Environment using python.
+ Install necessary libraries 
+ Train Machine Learning model
+ Get predictions on the test data 

## Project Instructions:

### This project can be run locally by following the steps:

+ Create Virtual Environmnet
```
python3 -m venv venv
```
+ Activate Virtual Environment

```
source venv/bin/activate
```
+ Install necessary libraries using pip

```
pip install -r requirements.txt 
```
```
+ Train the model and get predictions
```
python3  app/src/train.py
```
+ Get predictions of the test data, pass the path of test dataset ane make sure it has column names similar to that of training data,
pass False as the 2nd parameter if there is no target field in the test csv file other wise leave it 
```
python3  app/src/inference.py path_of_test_data 
```

### Running the project using docker  

+ Clone the repository 

+ Build Docker Image
```
docker build . -t ml_project
```
+ Run Docker Image
```
docker run  ml_project
```

+ Save Docker Image
```
docker save --output saved-image.tar ml_project
```

+ Load Docker Image
```
docker load --input saved-image.tar
```
