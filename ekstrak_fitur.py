import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
import pandas as pd

# Sekarang 'feature_stack_path' menunjuk ke file baru Anda
feature_stack_path = 'Hasil/feature_stack_training.tif'
training_data_path = 'data_latih.shp'

X_data = [] # Untuk menyimpan fitur (13 kolom)
y_data = [] # Untuk menyimpan label (1 kolom)

gdf = gpd.read_file(training_data_path)

print("Memulai Ekstraksi...")
with rasterio.open(feature_stack_path) as src:
    for index, row in gdf.iterrows():
        geometry = [row.geometry]
        label = row['label'] # 1=Padi, 2=Gulma
        
        # Potong tumpukan 13-band menggunakan poligon
        out_image, out_transform = mask(src, geometry, crop=True, nodata=np.nan)
        
        # Pertama, temukan piksel yang valid
        # Kita bisa gunakan band pertama sebagai referensi
        valid_mask_2d = ~np.isnan(out_image[0])
        
        # Siapkan data untuk poligon ini
        pixel_features = []
        for band_idx in range(src.count): 
            band_data = out_image[band_idx]
            valid_pixels = band_data[valid_mask_2d]
            pixel_features.append(valid_pixels)

        # Transpose array
        pixels_as_features = np.array(pixel_features).T # (jumlah_piksel, 13)
        
        # Buat label
        labels_for_pixels = np.full(pixels_as_features.shape[0], label)
        
        X_data.append(pixels_as_features)
        y_data.append(labels_for_pixels)

# Gabungkan semua data
X = np.concatenate(X_data, axis=0) # (total_piksel, 13 fitur)
y = np.concatenate(y_data, axis=0) # (total_piksel, label)

print(f"Ekstraksi selesai. Total piksel untuk dilatih: {X.shape[0]}")

print("Menyimpan data latih ke CSV...")
feature_names = ['red',
                 'green',
                 'blue',
                 'm_green',
                 'm_red',
                 'red_edge',
                 'nir',
                 'gndvi',
                 'ndrei',
                 'ndvi',
                 'savi'
                #  'idm',
                #  'contr'
]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y
output_csv_path = 'Hasil/data_training.csv'
df.to_csv(output_csv_path, index=False)

print(f"Data latih berhasil disimpan di: {output_csv_path}")