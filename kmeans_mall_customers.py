# ============================================================
# K-Means Clustering - Segmentasi Pelanggan Mall
# Dataset: Mall Customers (Annual Income & Spending Score)
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. PENGUMPULAN DAN PERSIAPAN DATA
# ============================================================

# Load dataset
baca = pd.read_csv("data/mall_customers.csv")
print("=== 5 Data Pertama ===")
print(baca.head())

# Informasi dataset
print("\n=== Informasi Dataset ===")
baca.info()

# Statistik deskriptif
print("\n=== Statistik Deskriptif ===")
print(baca.describe())

# ============================================================
# 2. PREPROCESSING DATA
# ============================================================

# Drop kolom yang tidak diperlukan untuk clustering
baca = baca.drop(["CustomerID", "Gender"], axis=1)
print("\n=== Data setelah drop kolom tidak relevan ===")
print(baca.head())

# Menentukan variabel yang akan diklusterkan
baca_x = baca[["Annual_Income_k", "Spending_Score"]]
print("\n=== Variabel untuk Clustering ===")
print(baca_x.head())

# Visualisasi persebaran data awal
plt.figure(figsize=(8, 6))
plt.scatter(baca["Annual_Income_k"], baca["Spending_Score"],
            s=50, c="red", marker="o", alpha=0.5)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Sebaran Data Pelanggan Mall (Sebelum Clustering)")
plt.tight_layout()
plt.savefig("images/scatter_awal.png", dpi=150)
plt.show()
print("Plot sebaran data awal disimpan.")

# ============================================================
# 3. NORMALISASI DATA (Min-Max Scaling)
# ============================================================

x_array = np.array(baca_x)
print("\n=== Array Data ===")
print(x_array[:5])

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
print("\n=== Data setelah Normalisasi (5 baris pertama) ===")
print(x_scaled[:5])

# ============================================================
# 4. MENENTUKAN JUMLAH CLUSTER OPTIMAL (Elbow Method)
# ============================================================

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(x_scaled)
    inertia.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, "bo-", linewidth=2, markersize=8)
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method - Menentukan Nilai K Optimal")
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/elbow_method.png", dpi=150)
plt.show()
print("Elbow method plot disimpan.")

# ============================================================
# 5. MEMBUAT MODEL K-MEANS (K=5)
# ============================================================

# Membuat model KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Training model
kmeans.fit(x_scaled)

# Menampilkan label cluster
print("\n=== Label Cluster ===")
print(kmeans.labels_)

# Menambahkan hasil cluster ke dataframe
baca["kluster"] = kmeans.labels_

# Menampilkan data dengan hasil cluster
print("\n=== Data dengan Hasil Cluster ===")
print(baca.head(10))

# Distribusi data per cluster
print("\n=== Distribusi Data per Cluster ===")
print(baca["kluster"].value_counts().sort_index())

# ============================================================
# 6. VISUALISASI HASIL CLUSTERING
# ============================================================

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
cluster_names = {
    0: "Low Income, High Spend",
    1: "Mid Income, Mid Spend",
    2: "Low Income, Low Spend",
    3: "High Income, High Spend",
    4: "High Income, Low Spend",
}

plt.figure(figsize=(10, 7))

for i in range(5):
    mask = baca["kluster"] == i
    plt.scatter(
        baca.loc[mask, "Annual_Income_k"],
        baca.loc[mask, "Spending_Score"],
        s=80,
        c=colors[i],
        label=f"Cluster {i}: {cluster_names[i]}",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )

# Titik centroid (denormalisasi balik ke skala asli)
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers_orig[:, 0],
    centers_orig[:, 1],
    s=300,
    c="black",
    marker="X",
    label="Centroid",
    zorder=5,
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering - Segmentasi Pelanggan Mall (K=5)")
plt.legend(loc="upper left", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/kmeans_hasil.png", dpi=150)
plt.show()
print("Visualisasi hasil clustering disimpan.")

# ============================================================
# 7. ANALISIS DAN LAPORAN HASIL CLUSTERING
# ============================================================

print("\n=== Analisis Rata-rata Tiap Cluster ===")
cluster_analysis = baca.groupby("kluster")[
    ["Age", "Annual_Income_k", "Spending_Score"]
].mean().round(2)
cluster_analysis.index.name = "Cluster"
print(cluster_analysis)

# Heatmap analisis cluster
plt.figure(figsize=(8, 5))
sns.heatmap(cluster_analysis, annot=True, fmt=".1f",
            cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Nilai Rata-rata"})
plt.title("Rata-rata Fitur per Cluster")
plt.tight_layout()
plt.savefig("images/cluster_heatmap.png", dpi=150)
plt.show()
print("Heatmap cluster disimpan.")

# Inertia / WCSS
print(f"\n=== Inertia (WCSS) Final: {kmeans.inertia_:.4f} ===")

# Posisi centroid
print("\n=== Posisi Centroid (Skala Asli) ===")
centroid_df = pd.DataFrame(centers_orig,
                            columns=["Annual_Income_k", "Spending_Score"])
centroid_df.index.name = "Cluster"
print(centroid_df.round(2))

print("\n============================")
print("PROSES CLUSTERING SELESAI!")
print("============================")
