import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

train_file_path = '../train.csv'
dataset_df = pd.read_csv(train_file_path)
print('Train dataset shape: {}'.format(dataset_df.shape))

print(dataset_df.head())

print(dataset_df.info())