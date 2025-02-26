import streamlit as st
from exif import Image as ExifImage
from haversine import haversine
import io

st.title("🛰️ Aplikasi Hitung Jarak dari Foto")
st.write("Upload dua gambar berisi informasi GPS untuk menghitung jarak")

def extract_coordinates(image_bytes):
    try:
        img = ExifImage(image_bytes)
    except Exception as e:
        st.error(f"Gagal membaca gambar: {str(e)}")
        return None
    
    if not img.has_exif:
        st.error("Gambar tidak memiliki metadata EXIF")
        return None
        
    try:
        lat = img.gps_latitude
        lat_ref = img.gps_latitude_ref
        lon = img.gps_longitude
        lon_ref = img.gps_longitude_ref
    except AttributeError:
        st.error("Gambar tidak memiliki informasi GPS")
        return None
    
    # Konversi koordinat ke desimal
    lat_decimal = lat[0] + lat[1]/60 + lat[2]/3600
    if lat_ref != 'N':
        lat_decimal = -lat_decimal
        
    lon_decimal = lon[0] + lon[1]/60 + lon[2]/3600
    if lon_ref != 'E':
        lon_decimal = -lon_decimal
        
    return (lat_decimal, lon_decimal)

# Upload gambar pertama
img1 = st.file_uploader("Upload Gambar 1", type=['jpg', 'jpeg'])
coord1 = None
if img1:
    bytes_data = img1.getvalue()
    st.image(bytes_data, caption="Gambar 1", use_column_width=True)
    coord1 = extract_coordinates(bytes_data)
    
    if coord1:
        st.success(f"**Koordinat 1:** {coord1[0]:.6f}°N, {coord1[1]:.6f}°E")

# Upload gambar kedua
img2 = st.file_uploader("Upload Gambar 2", type=['jpg', 'jpeg'])
coord2 = None
if img2:
    bytes_data = img2.getvalue()
    st.image(bytes_data, caption="Gambar 2", use_column_width=True)
    coord2 = extract_coordinates(bytes_data)
    
    if coord2:
        st.success(f"**Koordinat 2:** {coord2[0]:.6f}°N, {coord2[1]:.6f}°E")

# Hitung jarak jika kedua koordinat tersedia
if coord1 and coord2:
    try:
        distance = haversine(coord1, coord2)
        st.subheader(f"Jarak antara kedua lokasi: {distance:.2f} km")
        st.balloons()
    except Exception as e:
        st.error(f"Error menghitung jarak: {str(e)}")
elif coord1 or coord2:
    st.warning("Silakan upload kedua gambar untuk menghitung jarak")
