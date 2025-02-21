import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset of Penguins
df = pd.read_csv("penguins.csv", usecols=["species", "bill_length_mm", "bill_depth_mm"]).dropna()
X = df[["bill_length_mm", "bill_depth_mm"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Map clusters to species
cluster_mapping = {0: "Adelie", 1: "Gentoo", 2: "Chinstrap"}
df["species_cluster"] = df["cluster"].map(cluster_mapping)

# Visualize the clusters in an XY plane, like the figure below but with the result of your mode. .
colors = {"Adelie": "blue", "Gentoo": "orange", "Chinstrap": "green"}
for species, color in colors.items():
    subset = df[df["species_cluster"] == species]
    plt.scatter(subset["bill_length_mm"], subset["bill_depth_mm"], label=species, color=color)

# Put the "centroids" of each cluster in the figure
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", label="Centroids")

plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title("Penguin Clustering with K-Means")
plt.legend(loc='upper right')
plt.show()

# Evaluate the model and find the accuracy of your model
accuracy = (df["species"] == df["species_cluster"]).mean() * 100
print(f"K-Means Clustering Accuracy: {accuracy:.2f}%")
