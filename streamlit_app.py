import streamlit as st
from exif import Image as ExifImage
import io
from PIL import Image
import pillow_heif
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import base64
import simplekml
import requests
import os
from datetime import datetime

# Judul aplikasi
st.title("📸 Aplikasi Analisis Lokasi Foto Kronologis dengan Rute Aktual")

# Konversi HEIC ke JPEG
def convert_heic_to_jpeg(image_bytes):
    try:
        heif_file = pillow_heif.read_heif(image_bytes)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=90)
        return img_byte_arr.getvalue()
    except Exception as e:
        st.error(f"Gagal konversi HEIC: {str(e)}")
        return None

# Ekstrak koordinat GPS dari gambar
def extract_coordinates(image_bytes):
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except:
        return None, None
    if image_bytes.startswith(b"ftypheic"):
        image_bytes = convert_heic_to_jpeg(image_bytes)
        if not image_bytes:
            return None, None
    try:
        img = ExifImage(image_bytes)
    except:
        return None, None
    if not hasattr(img, 'gps_latitude'):
        return None, None
    lat = img.gps_latitude
    lon = img.gps_longitude
    lat_ref = img.gps_latitude_ref
    lon_ref = img.gps_longitude_ref
    lat_decimal = lat[0] + lat[1]/60 + lat[2]/3600
    if lat_ref != 'N':
        lat_decimal = -lat_decimal
    lon_decimal = lon[0] + lon[1]/60 + lon[2]/3600
    if lon_ref != 'E':
        lon_decimal = -lon_decimal
    return (lat_decimal, lon_decimal), image_bytes

# Hitung jarak dan durasi menggunakan OSRM Public Demo Server
def calculate_osrm_route(origin, destination):
    try:
        # Format koordinat untuk OSRM (longitude,latitude)
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        url = f"http://router.project-osrm.org/route/v1/driving/{coords}"
        
        # Kirim permintaan ke OSRM API
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 'Ok':
                # Ambil jarak (dalam meter) dan durasi (dalam detik)
                distance = data['routes'][0]['distance']  # Jarak dalam meter
                duration = data['routes'][0]['duration']  # Durasi dalam detik
                return distance, duration
            else:
                st.error("Gagal mendapatkan rute. Pastikan koordinat valid.")
                return None, None
        else:
            st.error(f"Error: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        return None, None

# Upload multiple gambar
uploaded_files = st.file_uploader(
    "Upload Foto (JPG/HEIC)",
    type=['jpg', 'jpeg', 'heic'],
    accept_multiple_files=True
)

# Urutkan file berdasarkan waktu pembuatan/modifikasi
if uploaded_files:
    def get_file_timestamp(file):
        try:
            # Coba ekstrak timestamp dari nama file (format YYYYMMDD_HHMMSS)
            if '_' in file.name:
                timestamp_str = file.name.split('.')[0]
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            pass
        
        # Fallback ke waktu upload (Streamlit UploadedFile tidak memiliki path fisik)
        return datetime.now()

    # Urutkan file berdasarkan timestamp
    sorted_files = sorted(uploaded_files, key=get_file_timestamp, reverse=False)
else:
    sorted_files = []

# Proses gambar
coordinates = []
thumbnails = []
valid_images = []
valid_data = []

for uploaded_file in sorted_files:
    bytes_data = uploaded_file.getvalue()
    coords, converted_bytes = extract_coordinates(bytes_data)
    
    if coords:
        # Simpan data
        coordinates.append(coords)
        valid_images.append(uploaded_file.name)
        
        # Buat thumbnail
        img = Image.open(io.BytesIO(converted_bytes))
        img.thumbnail((150, 150))
        thumbnails.append(img)
        
        # Simpan data untuk Excel
        valid_data.append({
            "Nama File": uploaded_file.name,
            "Latitude": coords[0],
            "Longitude": coords[1]
        })
    else:
        st.warning(f"Foto {uploaded_file.name} tidak memiliki data GPS")

# Tampilkan hasil
if coordinates:
    # Tetapkan titik pertama sebagai referensi (titik nol)
    reference_point = coordinates[0]
    
    # Hitung jarak rute aktual menggunakan OSRM
    distances = []
    for coord, filename in zip(coordinates, valid_images):
        distance, duration = calculate_osrm_route(reference_point, coord)
        if distance is not None and duration is not None:
            distances.append({
                "Nama File": filename,
                "Latitude": coord[0],
                "Longitude": coord[1],
                "Jarak (Meter)": round(distance, 2),
                "Durasi (Detik)": round(duration, 2)
            })
        else:
            st.warning(f"Gagal menghitung rute untuk foto {filename}.")
    
    # Urutkan data berdasarkan jarak dari terdekat ke terjauh
    sorted_distances = sorted(distances, key=lambda x: x["Jarak (Meter)"])
    
    # Tampilkan tabel hasil
    st.subheader("📊 Data Foto Berdasarkan Jarak Rute dari Titik Nol")
    df_sorted = pd.DataFrame(sorted_distances)
    st.dataframe(df_sorted.style.format({"Jarak (Meter)": "{:.2f}", "Durasi (Detik)": "{:.2f}"}))
    
    # Tampilkan thumbnail
    st.subheader("📸 Thumbnail Foto (Urutan Jarak)")
    cols = st.columns(4)
    for idx, row in enumerate(sorted_distances):
        with cols[idx % 4]:
            thumbnail = thumbnails[valid_images.index(row["Nama File"])]
            st.image(thumbnail, caption=row["Nama File"], use_column_width=True)
    
    # Buat peta dengan marker diurutkan berdasarkan jarak
    m = folium.Map(location=reference_point, zoom_start=14)
    marker_cluster = MarkerCluster().add_to(m)
    
    for row in sorted_distances:
        coord = (row["Latitude"], row["Longitude"])
        filename = row["Nama File"]
        
        # Konversi gambar ke base64 untuk popup
        thumbnail = thumbnails[valid_images.index(filename)]
        img_byte = io.BytesIO()
        thumbnail.save(img_byte, format='PNG')
        encoded = base64.b64encode(img_byte.getvalue()).decode()
        html = f'<img src="data:image/png;base64,{encoded}" width="150"/><br>{filename}'
        
        folium.Marker(
            location=coord,
            popup=html,
            tooltip=filename
        ).add_to(marker_cluster)
    
    # Tampilkan peta
    st.subheader("📍 Peta Lokasi (Urutan Jarak)")
    st_folium(m, width=700, height=500)
    
    # Download Excel
    df_excel = pd.DataFrame(sorted_distances)
    excel_file = io.BytesIO()
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_excel.to_excel(writer, index=False)
    excel_file.seek(0)
    
    st.download_button(
        label="📥 Download Excel",
        data=excel_file,
        file_name="data_foto.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Download KML
    kml = simplekml.Kml()
    for row in sorted_distances:
        coord = (row["Latitude"], row["Longitude"])
        filename = row["Nama File"]
        kml.newpoint(
            name=filename,
            coords=[(coord[1], coord[0])]
        )
    kml_file = kml.kml().encode('utf-8')
    
    st.download_button(
        label="📥 Download KML",
        data=kml_file,
        file_name="lokasi_foto.kml",
        mime="application/vnd.google-earth.kml+xml"
    )
else:
    st.info("Silakan upload foto dengan data GPS")
