import rasterio
import numpy as np

folder_path = r''
homogeneity_path = r''
contrast_path = r''
print("Menumpuk seluruh fitur...")

# Daftar file yang ingin digabungkan
# Pastikan urutannya benar!
files_to_stack = [
    rf'{folder_path}/Cikembar_30_clip.tif', 
    rf'{folder_path}/GNDVI.tif', 
    rf'{folder_path}/NDREI.tif', 
    rf'{folder_path}/NDVI.tif', 
    rf'{folder_path}/SAVI.tif'
    # rf'{folder_path}/{homogeneity_path}',
    # rf'{folder_path}/{contrast_path}'
]

# Baca semua file dan kumpulkan datanya
data_stack = []
profile = None # Kita akan ambil metadata dari file pertama

# Loop pertama: baca semua data
for files in files_to_stack:
    with rasterio.open(files) as src:
        if profile is None:
            profile = src.profile
        
        # Baca semua band dari file ini
        data_stack.append(src.read())

# Gabungkan semua data menjadi satu array NumPy besar
full_stack_array = np.vstack(data_stack)

# Perbarui profile untuk file stack baru
total_bands = full_stack_array.shape[0]
profile.update(count=total_bands, nodata=np.nan) 

# Tulis ke file stack baru
output_stack_path = 'Hasil/feature_stack_training.tif'
with rasterio.open(output_stack_path, 'w', **profile) as dst:
    dst.write(full_stack_array)

print(f"Selesai! Tumpukan fitur dengan {total_bands} band disimpan di: {output_stack_path}")