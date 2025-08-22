#Importing all the libraries
import pandas as pd
from pycaret.classification import *

#Load Dataset
data= pd.read_csv('Dataset\\train.csv')
print(f"Dataset Shape : {data.shape}")

# Columns to ignore (non-predictive or post-churn info)
ignore_features = [
    'Customer ID',
    'Churn Category',
    'Churn Reason',
    'Churn Score',
    'City',
    'Country',
    'Customer Status',
    'Lat Long',
    'Latitude',
    'Longitude',
    'Population',
    'Quarter',
    'State',
    'Zip Code'
]
#Initialize Pycaret
clf_setup = setup(data=data, target='Churn', session_id=123,normalize=True, ignore_features=ignore_features, verbose=True)

#compare_models
best_model=compare_models()

#Finalize model
final_model=finalize_model(best_model)

#Evaluate model
evaluate_model(final_model)

#Save model
save_model(final_model,'churn_model')

print("\n Model Training complete!")