import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import resample

input_data = 'syn_input/DeDupeTrain.csv'
data = pd.read_csv(input_data)
print(data.shape)


ham_messages = data[data["Dupe"] == 1]
spam_messages  = data[data["Dupe"] == 0]
print(ham_messages.head())
print(ham_messages.shape)
print(spam_messages.head())
print(spam_messages.shape)
print("after sampling")
spam_upsample = resample(data,replace=True,n_samples=500,random_state=42)
print(spam_upsample.shape)
print(spam_upsample.head())
spam_upsample.to_csv("tmp.csv",index=False)
