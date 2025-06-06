import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('Mall_Customers.csv')


X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)



from sklearn.metrics import silhouette_score, davies_bouldin_score

sil_score = silhouette_score(X, y_kmeans)
db_score = davies_bouldin_score(X, y_kmeans)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Index: {db_score:.3f}")

# Add cluster labels to the DataFrame
df['Cluster'] = y_kmeans


plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(
        X.values[y_kmeans == i, 0],
        X.values[y_kmeans == i, 1],
        s=50,
        c=colors[i],
        label=f'Cluster {i+1}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='yellow',
    label='Centroids',
    edgecolor='black'
)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()