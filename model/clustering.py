import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    selected_columns = [
        'Age',
        'Avg_Daily_Usage_Hours',
        'Sleep_Hours_Per_Night',
        'Mental_Health_Score',
        'Conflicts_Over_Social_Media',
        'Addicted_Score'
    ]

    missing = [col for col in selected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing}")

    df_selected = df[selected_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected)

    return df_selected, X_scaled

def plot_elbow(X_scaled, save_dir):
    inertia = []
    silhouette = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

    plt.figure()
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Inertia')
    plt.title('Elbow Method (Inertia vs K)')
    plt.savefig(os.path.join(save_dir, 'elbow_inertia.png'))

    plt.figure()
    plt.plot(K_range, silhouette, 'ro-')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.savefig(os.path.join(save_dir, 'silhouette.png'))

def perform_clustering(X_scaled, df_original, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_clustered = df_original.copy()
    df_clustered['Cluster'] = labels
    return df_clustered, labels

def plot_cluster(df, labels, save_path):
    plt.figure()
    sns.scatterplot(data=df, x='Avg_Daily_Usage_Hours', y='Sleep_Hours_Per_Night', hue=labels, palette='Set2')
    plt.title('Cluster Plot (Usage vs Sleep)')
    plt.savefig(save_path)
