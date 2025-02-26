import streamlit as st
from exif import Image as ExifImage
from haversine import haversine
import io
from PIL import Image
import pillow_heif
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import base64
import json

st.title("🗺️ Aplikasi Analisis Lokasi Foto")

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

# Ekstrak koordinat
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

# Upload multiple gambar
uploaded_files = st.file_uploader(
    "Upload Foto (JPG/HEIC)",
    type=['jpg', 'jpeg', 'heic'],
    accept_multiple_files=True
)

# Proses gambar
coordinates = []
thumbnails = []
valid_images = []

for uploaded_file in uploaded_files:
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
    else:
        st.warning(f"Foto {uploaded_file.name} tidak memiliki data GPS")

# Tampilkan hasil
if coordinates:
    # Hitung jarak semua pasang
    num_images = len(coordinates)
    distance_matrix = [[0.0 for _ in range(num_images)] for _ in range(num_images)]
    
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                distance = haversine(coordinates[i], coordinates[j]) * 1000  # Konversi ke meter
                distance_matrix[i][j] = round(distance, 2)
    
    # Tampilkan tabel jarak
    st.subheader("📊 Matriks Jarak (Meter)")
    df = pd.DataFrame(
        distance_matrix,
        index=valid_images,
        columns=valid_images
    )
    st.dataframe(df.style.format("{:.2f}"))

    # Tampilkan thumbnail
    st.subheader("📸 Thumbnail Foto")
    cols = st.columns(4)
    for idx, (thumbnail, filename) in enumerate(zip(thumbnails, valid_images)):
        with cols[idx % 4]:
            st.image(thumbnail, caption=filename, use_column_width=True)

    # Buat peta
    m = folium.Map(location=coordinates[0], zoom_start=14)
    marker_cluster = MarkerCluster().add_to(m)
    
    for coord, filename, thumbnail in zip(coordinates, valid_images, thumbnails):
        # Konversi gambar ke base64 untuk popup
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
    st.subheader("📍 Peta Lokasi")
    st_folium(m, width=700, height=500)

    # Download GeoJSON
    features = []
    for coord, filename in zip(coordinates, valid_images):
        features.append({
            "type": "Feature",
            "properties": {
                "name": filename,
                "popupContent": f"<strong>{filename}</strong><br>Lat: {coord[0]:.6f}<br>Lon: {coord[1]:.6f}"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [coord[1], coord[0]]
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    st.download_button(
        label="📥 Download GeoJSON",
        data=json.dumps(geojson, indent=2),
        file_name="foto_lokasi.geojson",
        mime="application/json"
    )
else:
    st.info("Silakan upload foto dengan data GPS")
