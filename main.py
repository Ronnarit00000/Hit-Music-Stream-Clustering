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

PATH = os.environ.get("SPOTIFY_CSV", "data/spotify_2023.csv")
CHART_DIR = "chart"


def load_data(path: str):
    return pd.read_csv(path, encoding='latin-1')


# --- ลบ comma ก่อนแปลงเป็นตัวเลข (เช่น "1,170" -> 1170) ไม่เช่นนั้นจะกลายเป็น NaN -> 0 ---
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


class HitMusicStreamModel:
    def __init__(self, data_frame, x_scaled, k_range, features):
        self.data_frame = data_frame
        self.x_scaled = x_scaled
        self.k_range = k_range
        self.features = features
        self.k_cluster = None
        self.kmeans = None

    # --- หาค่า k ---
    def clustering_evaluation(self):
        elbow, silhouette = [], []
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.x_scaled)
            elbow.append(km.inertia_)
            silhouette.append(silhouette_score(self.x_scaled, labels))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot(self.k_range, elbow, 'o-', color='#1f77b4', linewidth=1.5, markersize=6)
        axes[0].set_title('Elbow Method (option k)', fontsize=12)
        axes[0].set_xlabel('number of cluster (k)', fontsize=10)
        axes[0].set_ylabel('inertia', fontsize=10)
        axes[0].set_xticks(self.k_range)

        axes[1].plot(self.k_range, silhouette, 'o-', color='#2ca02c', linewidth=1.5, markersize=6)
        axes[1].set_title('Silhouette Method (cluster separation)', fontsize=12)
        axes[1].set_xlabel('number of cluster (k)', fontsize=10)
        axes[1].set_ylabel('silhouette score', fontsize=10)
        axes[1].set_xticks(self.k_range)

        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/evaluation-{int(time.time())}.png', dpi=300)
        plt.close()

        best_k = list(self.k_range)[int(np.argmax(silhouette))]
        print(f"k ที่ silhouette สูงสุด = {best_k} ({max(silhouette)})")
        self.k_cluster = best_k
        return best_k

    # --- รัน K-Means ด้วยจำนวน cluster ที่เลือก แล้วเพิ่มคอลัมน์ 'cluster' ลงใน data_frame ---
    def fit_kmeans(self, k=None):
        if k is not None:
            self.k_cluster = k
        self.kmeans = KMeans(n_clusters=self.k_cluster, random_state=42, n_init=10)
        self.data_frame["cluster"] = self.kmeans.fit_predict(self.x_scaled)
        print("\nจำนวนเพลงในแต่ละ cluster:")
        print(self.data_frame["cluster"].value_counts().sort_index())

    # --- ค่าเฉลี่ยของแต่ละ feature ในแต่ละ cluster (ใช้ตีความว่า cluster ไหนคือเพลงฮิต) ---
    def cluster_summary(self):
        summary = self.data_frame.groupby("cluster")[self.features].mean().round(2)
        summary["n_songs"] = self.data_frame["cluster"].value_counts().sort_index()
        print("\nค่าเฉลี่ยของแต่ละ cluster:")
        print(summary)

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
        plt.savefig(f'{CHART_DIR}/clusters-{int(time.time())}.png', dpi=300)
        plt.close()

    # --- กราฟแท่งเปรียบเทียบ feature ของแต่ละ cluster (หลัง scale แล้ว อ่านเทียบกันได้) ---
    def plot_cluster_profile(self):
        scaled_df = pd.DataFrame(self.x_scaled, columns=self.features)
        scaled_df["cluster"] = self.data_frame["cluster"].values
        profile = scaled_df.groupby("cluster")[self.features].mean()

        ax = profile.T.plot(kind="bar", figsize=(12, 6), width=0.8, colormap='tab10')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title('Cluster Profile (standardized mean of each feature)', fontsize=12)
        ax.set_ylabel('z-score')
        ax.legend(title='cluster')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/profile-{int(time.time())}.png', dpi=300)
        plt.close()

    # --- กราฟแท่งจำนวนเพลงต่อ cluster ---
    def plot_cluster_size(self):
        counts = self.data_frame["cluster"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(counts.index.astype(str), counts.values, color=plt.cm.tab10.colors[:len(counts)])
        ax.bar_label(bars)
        ax.set_title('Number of songs per cluster', fontsize=12)
        ax.set_xlabel('cluster')
        ax.set_ylabel('songs')
        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/cluster-size-{int(time.time())}.png', dpi=300)
        plt.close()

    # --- ตัวอย่างเพลงยอดสตรีมสูงสุดในแต่ละ cluster ---
    def top_songs(self, n=5):
        for c, g in self.data_frame.groupby("cluster"):
            print(f"\n--- cluster {c} : top {n} by streams ---")
            print(g.nlargest(n, "streams")[["track_name", "artist(s)_name", "streams"]].to_string(index=False))


if __name__ == "__main__":
    os.makedirs(CHART_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ==============================================================================
    # 1. DATA PREPROCESSING
    # ==============================================================================

    FEATURES = [
        'streams',
        'in_spotify_charts',
        'in_apple_charts',
        'in_deezer_charts',
        'in_shazam_charts',
        'in_spotify_playlists',
        'in_apple_playlists'
    ]

    DATA_FRAME = clean_data(load_data(PATH))
    X_SCALED = find_x_scaled(DATA_FRAME, FEATURES)
    K_RANGE = range(2, 11)

    # ==============================================================================
    # 2. MODEL: CLUSTERING
    # ==============================================================================

    model = HitMusicStreamModel(DATA_FRAME, X_SCALED, K_RANGE, FEATURES)
    best_k = model.clustering_evaluation()      # 1) ดู elbow + silhouette เพื่อเลือก k
    model.fit_kmeans(best_k)                    # 2) แก้เลข k ตามที่เลือกจากกราฟ (หรือใส่ best_k)
    model.cluster_summary()                     # 3) สรุปค่าเฉลี่ยต่อ cluster
    model.plot_clusters()                       # 4) กราฟ scatter (PCA)
    model.plot_cluster_profile()                # 5) กราฟโปรไฟล์ของแต่ละ cluster
    model.plot_cluster_size()                   # 6) กราฟจำนวนเพลงต่อ cluster
    model.top_songs()                           # 7) ตัวอย่างเพลงในแต่ละ cluster