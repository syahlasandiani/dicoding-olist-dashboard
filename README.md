# Proyek Analisis Data: E-Commerce Public Dataset (Olist) ✨

Proyek akhir kelas analisis data Dicoding. Analisis mencakup tren penjualan, performa kategori produk,
performa wilayah dan logistik, serta segmentasi pelanggan (RFM) pada pesanan berstatus *delivered*
periode **Januari 2017 - Agustus 2018**.

## Struktur Direktori

```
submission
├───dashboard
| ├───main_data.csv
| └───dashboard.py
├───data
| ├───olist_customers_dataset.csv
| ├───olist_orders_dataset.csv
| ├───olist_order_items_dataset.csv
| ├───olist_products_dataset.csv
| ├───olist_geolocation_dataset.csv
| └───product_category_name_translation.csv
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt
```

## Setup Environment - Anaconda

```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```
mkdir proyek_analisis_data
cd proyek_analisis_data
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan Notebook

Buka `notebook.ipynb` dengan Jupyter Notebook / VS Code / Google Colab, lalu jalankan semua sel
(*Restart & Run All*). Notebook akan membuat ulang file `dashboard/main_data.csv`.

## Menjalankan Dashboard Streamlit

```
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka di `http://localhost:8501`.
