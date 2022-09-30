import common
import pyarrow
import graphviz
import IPython.display
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.tree
import yaml

common


df = pd.read_parquet("baskets-s.parquet")
print(df)
features = [
  "week",
    "customer",
    "product",
    "price",
]


target_names = df["product"].unique()
print(target_names)
X = df[features].values
y = df["product"].values

custom_list = df.values.tolist()

unique_list = []

for i in range(2000):
    unique_list.append([str(i)])

for entry in custom_list:
    add_index = int(entry[1])
    unique_list[add_index].append([entry[0], entry[-1]])

time_series = []

for i in range(89):
    time_series.append([str(i)])


cnt = Counter()
all_products = []
for entry in unique_list[88:89]:
    first_week = 0
    for x in range(0, 89):
        for n in entry[1:]:
            if str(x) == str(n[0]):
                time_series[x].append(n[1])
                print(time_series)

binary_matrix = []
product_matrix = []

for entry in time_series:
    for product in entry[1:]:
        if str(product) not in product_matrix:
            product_matrix.append([str(product)])

    print(product_matrix)

for entry in time_series:
    counter = 0
    for prod in product_matrix:
        for product in entry[1:]:
            print(prod[0],product)
            if str(prod[0]) == str(product):
                product_matrix[counter].append("1")
                print(product_matrix)
            else:
                product_matrix[counter].append("0")
        counter += 1


print(product_matrix)







