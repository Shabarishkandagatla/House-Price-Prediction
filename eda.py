import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/housing.csv")

plt.figure(figsize=(8,5))
plt.hist(df["median_house_value"], bins=30)
plt.title("House Price Distribution")
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.show()