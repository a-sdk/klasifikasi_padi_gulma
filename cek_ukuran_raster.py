import rasterio

def cek_ukuran_raster(input_raster):
    """
    Menghitung ukuran raster dan menampilkannya di terminal/shell.

    Args:
        input_raster (np.ndarray): Array NumPy yang akan dihitung.
        
    Returns:
        None.
    """
    with rasterio.open(input_raster) as src:
        print(f"Memeriksa bentuk...")
        band1 = src.read(1)
        print(f"Bentuk (shape) dari {input_raster} adalah {band1.shape}")

        
path = r""
cek_ukuran_raster(path)