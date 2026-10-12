# Hit Music Stream Clustering (Spotify 2023)

โปรเจกต์นี้ใช้ **KMeans Clustering** แบ่งกลุ่มเพลงจากชุดข้อมูล Spotify 2023 ตามความนิยมบนแพลตฟอร์มต่าง ๆ (ยอดสตรีม, อันดับในชาร์ต, จำนวน playlist) เพื่อดูว่าเพลงกลุ่มไหนเป็น "เพลงฮิต"

## สิ่งที่โค้ดทำ

1. โหลดและทำความสะอาดข้อมูล (แปลงคอลัมน์ตัวเลขที่เป็นข้อความ เช่น `"1,170"`)
2. Standardize ข้อมูล (StandardScaler)
3. หาจำนวน cluster ที่เหมาะสมด้วย **Elbow Method** และ **Silhouette Score**
4. รัน **KMeans** ด้วยจำนวน cluster ที่เลือก
5. สรุปผลเป็นตารางและกราฟ แล้วบันทึกเป็นไฟล์

## โครงสร้างโฟลเดอร์

```
project/
├── main.py                    # โค้ดหลัก
├── README.md
├── data/
│   └── spotify_2023.csv       # ไฟล์ข้อมูล (ต้องวางเอง)
└── chart/                     # สร้างอัตโนมัติเมื่อรันโค้ด
    ├── evaluation-<time>.png
    ├── clusters-<time>.png
```

## สิ่งที่ต้องมี

- Python 3.9 ขึ้นไป (ใช้ type hint แบบ `list[str]`)
- ไลบรารี: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

ติดตั้ง:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## วิธีใช้งาน

1. วางไฟล์ `spotify_2023.csv` ไว้ในโฟลเดอร์ `data/`
2. รันโค้ด:

```bash
python hit_music_kmeans.py
```

หรือถ้าไฟล์ข้อมูลอยู่ที่อื่น ให้กำหนด path ผ่าน environment variable:

```bash
# macOS / Linux
SPOTIFY_CSV=/path/to/spotify_2023.csv python hit_music_kmeans.py

# Windows (PowerShell)
$env:SPOTIFY_CSV="C:\path\to\spotify_2023.csv"; python hit_music_kmeans.py
```

### ปรับจำนวน cluster

ดูกราฟ `evaluation-*.png` แล้วเลือก k ที่เหมาะสม จากนั้นแก้บรรทัดนี้ใน `__main__`:

```python
model.fit_kmeans(k=3)        # เปลี่ยนเลข 3 เป็นค่าที่ต้องการ
# model.fit_kmeans(k=best_k) # หรือใช้ค่าที่ silhouette สูงสุดอัตโนมัติ
```

## อธิบายโค้ด

### ฟังก์ชัน

| ฟังก์ชัน | หน้าที่ |
|---|---|
| `load_data(path)` | อ่านไฟล์ CSV ด้วย encoding `latin-1` (ไฟล์มีชื่อเพลง/ศิลปินที่มีอักขระพิเศษ) |
| `to_number(series)` | ลบเครื่องหมาย `,` ก่อน แล้วแปลงเป็นตัวเลข ค่าที่แปลงไม่ได้จะเป็น 0 |
| `clean_data(data)` | ทำความสะอาดคอลัมน์ตัวเลข (`streams`, `key`, `in_shazam_charts`, `in_deezer_charts`, `in_spotify_playlists`, `in_apple_playlists`) |
| `find_x_scaled(data, features)` | ทำ standardization ให้ทุก feature มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1 เพื่อไม่ให้ feature ที่ตัวเลขใหญ่ (เช่น streams) ครอบงำการคำนวณระยะทาง |

> หมายเหตุ: ต้องลบ comma **ก่อน** แปลงเป็นตัวเลข เช่น `in_shazam_charts` มีค่าอย่าง `"1,170"` ถ้าแปลงก่อนจะกลายเป็น NaN และถูกแทนที่ด้วย 0 ทำให้ข้อมูลผิด

### Features ที่ใช้จัดกลุ่ม

`streams`, `in_spotify_charts`, `in_apple_charts`, `in_deezer_charts`, `in_shazam_charts`, `in_spotify_playlists`, `in_apple_playlists`

### Class `HitMusicStreamModel`

สร้างด้วย `HitMusicStreamModel(data_frame, x_scaled, k_range, features)`

| Method | หน้าที่ |
|---|---|
| `clustering_evaluation()` | รัน KMeans สำหรับทุกค่า k ใน `k_range` คำนวณ inertia (Elbow) และ Silhouette Score บันทึกกราฟเปรียบเทียบ และ return ค่า k ที่ silhouette สูงสุด |
| `fit_kmeans(k)` | รัน KMeans ด้วยค่า k ที่เลือก เพิ่มคอลัมน์ `cluster` ลงใน DataFrame และพิมพ์จำนวนเพลงในแต่ละ cluster |
| `plot_clusters()` | ลดมิติด้วย PCA เหลือ 2 แกน แล้ววาด scatter plot สีตาม cluster พร้อมจุด centroid |

## ผลลัพธ์ที่ได้

### ไฟล์กราฟ (ในโฟลเดอร์ `chart/`)

- `evaluation-*.png` กราฟ Elbow และ Silhouette ใช้เลือกค่า k
- `clusters-*.png` scatter plot ของ cluster (PCA 2D)
