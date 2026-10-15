import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def load_data(path: str):
    return pd.read_csv(path, encoding='latin-1')


def to_number(series: pd.Series):
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce").fillna(0)


def clean_data(data: pd.DataFrame):
    lst_to_clean = [
        "key",
        "streams",
        "in_shazam_charts",
        "in_deezer_charts",
        "in_spotify_playlists",
        "in_apple_playlists"
    ]

    for col in lst_to_clean:
        data[col] = to_number(data[col])
    return data


def find_x_scaled(data: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    return scaler.fit_transform(data[features])


class KMeansClustering:
    def __init__(self, data_frame, x_scaled, k_range, features, chart_directory):
        self.chart_dir = chart_directory
        self.data_frame = data_frame
        self.features = features
        self.x_scaled = x_scaled
        self.k_range = k_range
        self.k_cluster = None
        self.kmeans = None
        self.elbow = None
        self.silhouette = None

    # --- หาค่า k ---
    def evaluation(self):
        elbow, silhouette = [], []
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.x_scaled)
            elbow.append(km.inertia_)
            silhouette.append(silhouette_score(self.x_scaled, labels))

        self.elbow = elbow
        self.silhouette = silhouette

    # --- รัน K-Means ด้วยจำนวน cluster ที่เลือก แล้วเพิ่มคอลัมน์ 'cluster' ลงใน data_frame ---
    def fit(self, k: int):
        self.k_cluster = k
        self.kmeans = KMeans(n_clusters=self.k_cluster, random_state=42, n_init=10)
        self.data_frame["cluster"] = self.kmeans.fit_predict(self.x_scaled)

    # --- สร้าง plot elbow กับ silhouette เพื่อหา best K
    def plot_evaluation(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(self.k_range, self.elbow, 'o-', color='#1f77b4', linewidth=1.5, markersize=6)
        axes[0].set_title('Elbow Method (option k)', fontsize=12)
        axes[0].set_xlabel('number of cluster (k)', fontsize=10)
        axes[0].set_ylabel('inertia', fontsize=10)
        axes[0].set_xticks(self.k_range)

        axes[1].plot(self.k_range, self.silhouette, 'o-', color='#2ca02c', linewidth=1.5, markersize=6)
        axes[1].set_title('Silhouette Method (cluster separation)', fontsize=12)
        axes[1].set_xlabel('number of cluster (k)', fontsize=10)
        axes[1].set_ylabel('silhouette score', fontsize=10)
        axes[1].set_xticks(self.k_range)

        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/evaluation-{int(time.time())}.png', dpi=300)
        plt.close()

    # --- Scatter plot ของ cluster (ลดมิติด้วย PCA เหลือ 2 แกน) + จุดศูนย์กลาง ---
    def plot_clusters(self):
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.x_scaled)
        centers = pca.transform(self.kmeans.cluster_centers_)
        labels = self.data_frame["cluster"].values

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(coords[:, 0],  coords[:, 1],  c=labels,  cmap='tab10', s=18,  alpha=0.7)
        ax.scatter(centers[:, 0], centers[:, 1],  c='black', marker='X', s=200, edgecolors='white', linewidths=1.5, label='centroid')
        ax.set_title(f'K-Means Clusters (k={self.k_cluster}) - PCA 2D', fontsize=12)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        handles, _ = scatter.legend_elements()
        ax.legend(handles + ax.get_legend_handles_labels()[0], [f'cluster {i}' for i in range(self.k_cluster)] + ['centroid'])

        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/clusters-{int(time.time())}.png', dpi=300)
        plt.close()

    def plot_cluster_size(self):
        counts = self.data_frame["cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(counts.index.astype(str), counts.values, color=plt.cm.tab10.colors[:len(counts)])
        ax.bar_label(bars)
        ax.set_title('Number of songs per cluster', fontsize=12)
        ax.set_xlabel('cluster')
        ax.set_ylabel('songs')
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/cluster-size-{int(time.time())}.png', dpi=300)
        plt.close()


if __name__ == "__main__":
    # กำหนด folder ไว้เก็บ chart
    CHART_DIRECTORY = 'chart'

    # path dataset
    PATH = os.environ.get("SPOTIFY_CSV", "data/spotify_2023.csv")

    os.makedirs(CHART_DIRECTORY, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ==============================================================================
    # 1. DATA PREPROCESSING
    # ==============================================================================

    # load data
    DATA = load_data(PATH)

    # clean data
    DATA_FRAME = clean_data(DATA)

    # เลือก feature ที่ใช่ในการจัดกลุม
    FEATURES = [
        'streams',
        'in_spotify_charts',
        'in_apple_charts',
        'in_deezer_charts',
        'in_shazam_charts',
        'in_spotify_playlists',
        'in_apple_playlists'
    ]

    # ทำ standardization ให้ทุก feature มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1 เพื่อไม่ให้ feature ที่ตัวเลขใหญ่
    X_SCALED = find_x_scaled(DATA_FRAME, FEATURES)

    # กำหนด range การจัดกลุ่ม 2 - 10
    K_RANGE = range(2, 11)

    # ==============================================================================
    # 2. MODEL: CLUSTERING
    # ==============================================================================

    model = KMeansClustering(DATA_FRAME, X_SCALED, K_RANGE, FEATURES, CHART_DIRECTORY)

    # รัน KMeans สำหรับทุกค่า k ใน k_range คำนวณ inertia (Elbow) และ Silhouette Score บันทึกกราฟเปรียบเทียบ
    model.evaluation()

    # ดู elbow + silhouette จากรูป evaluation-*.png เพื่อเลือก k
    model.plot_evaluation()

    # การจัดกลุ่ม ตามที่เลือกจากกราฟ (หรือใส่ best_k)
    BEST_K = 3

    # รัน KMeans ด้วยค่า k ที่เลือก เพิ่มคอลัมน์ cluster ลงใน DataFrame และพิมพ์จำนวนเพลงในแต่ละ cluster
    model.fit(BEST_K)

    # กราฟ scatter ลดมิติด้วย PCA เหลือ 2 แกน แล้ววาด scatter plot สีตาม cluster พร้อมจุด centroid
    model.plot_clusters()

    # กราฟ แสดงเพลงใน
    model.plot_cluster_size()