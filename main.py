import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logReg import logRegModel

# Import training and test data
train = pd.read_csv(r"Data\test - train.csv")
test = pd.read_csv(r"Data\test - test.csv")

# Train model with training data
model = logRegModel(train)
model.trainModel(alpha=0.1, num_iters=1000, lam=0.1)

# Plot training data and boundary
model.plot2dProjection(i=1, j=2)

# Test model on test data
testModel, stats = model.test(test)

# Print statistical overview
print(stats)

# Plot test data and boundary
testModel.plot2dProjection()

