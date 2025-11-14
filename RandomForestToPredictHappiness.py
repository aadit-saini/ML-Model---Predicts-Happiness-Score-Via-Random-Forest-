import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("city_lifestyle_dataset.csv")

y = data['happiness_score']
features = ['population_density','avg_income','internet_penetration','avg_rent','air_quality_index','green_space_ratio','public_transport_score']
train_x,test_x,train_y,test_y = train_test_split(data[features],y,test_size=0.2,random_state=42)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_x, train_y)
predictions = model.predict(test_x)

r2 = r2_score(test_y, predictions)
print(f"R² score: {r2}")