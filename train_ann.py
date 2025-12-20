# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 00:28:51 2025

@author: Admin
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("data/diabetes_standard.csv")

FEATURE_NAMES = [
    "Age","DelayedHealing","SuddenWeightLoss","VisualBlurring",
    "Obesity","Polyphagia","Polyuria","Gender","Polydipsia","Irritability"
]

X = df[FEATURE_NAMES].values
y = df["class"].values

# ANN model
model = Sequential([
    Dense(32, activation="relu", input_shape=(10,)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Save model
model.save("Model/ann_diabetes.h5")

print("✅ ANN model saved successfully")
