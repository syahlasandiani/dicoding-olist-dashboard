import io
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------
# Konfigurasi halaman, tema, dan CSS
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Olist E-Commerce Dashboard", layout="wide")

NAVY = "#12263A"
TEAL = "#1f5fbf"
MUTED = "#b8c0cc"
WARN = "#d1495b"
TEXT = "#12263A"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans"],
    "svg.fonttype": "none",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "axes.grid": False,
    "font.size": 10,
})

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, .stApp, .stApp *, [data-testid="stSidebar"] *, svg text {
        font-family: 'Inter', sans-serif !important;
    }
    /* ikon bawaan Streamlit tetap memakai font ikonnya */
    [data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "main_data.csv"


# ----------------------------------------------------------------------------
# Helper visual
# ----------------------------------------------------------------------------
def money_fmt(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:,.1f} jt"
    return f"{v / 1e3:,.0f} rb"


def new_fig(w=7.5, h=4.2):
    return plt.subplots(figsize=(w, h))


def finish(ax, title, xlabel="", ylabel="", grid_y=False):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    if grid_y:
        ax.yaxis.grid(True, color=MUTED, alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)


def show(fig):
    """Tampilkan grafik sebagai SVG di dalam iframe yang memuat font Inter."""
    w, h = fig.get_size_inches()
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(
        r'\swidth="[^"]*"\s+height="[^"]*"',
        ' width="100%" height="100%" preserveAspectRatio="xMidYMin meet"',
        svg,
        count=1,
    )
    page = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');"
        "html,body{margin:0;padding:0;height:100%;background:transparent;}"
        "svg text{font-family:'Inter',sans-serif !important;}"
        "</style>" + svg
    )
    components.html(page, height=int(620 * h / w) + 8)
    plt.close(fig)


def insight(text):
    st.info(text)


# ----------------------------------------------------------------------------
# Load data & analisis RFM (sama dengan notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"],
    )


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(last_purchase=("order_purchase_timestamp", "max"),
             frequency=("order_id", "nunique"),
             monetary=("price", "sum"))
        .reset_index()
    )
    rfm["recency"] = (ref_date - rfm["last_purchase"]).dt.days

    if len(rfm) < 4:
        rfm["segment"] = "Hilang / Bernilai Rendah"
        return rfm

    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["f_score"] = np.select([rfm["frequency"] == 1, rfm["frequency"] == 2], [1, 2], default=3)

    def segment(row):
        if row["f_score"] >= 2 and row["r_score"] >= 3:
            return "Champions"
        if row["f_score"] >= 2:
            return "Loyal - Perlu Diaktifkan Kembali"
        if row["r_score"] >= 3 and row["m_score"] >= 3:
            return "Pelanggan Baru Bernilai Tinggi"
        if row["r_score"] >= 3:
            return "Pelanggan Baru"
        if row["m_score"] >= 3:
            return "Berisiko Hilang (Bernilai Tinggi)"
        return "Hilang / Bernilai Rendah"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


SEGMENT_ORDER = [
    "Champions",
    "Loyal - Perlu Diaktifkan Kembali",
    "Pelanggan Baru Bernilai Tinggi",
    "Pelanggan Baru",
    "Berisiko Hilang (Bernilai Tinggi)",
    "Hilang / Bernilai Rendah",
]

# ----------------------------------------------------------------------------
# Sidebar filter
# ----------------------------------------------------------------------------
data = load_data()

with st.sidebar:
    st.header("Filter Data")
    min_date = data["order_purchase_timestamp"].min().date()
    max_date = data["order_purchase_timestamp"].max().date()

    date_range = st.date_input(
        "Rentang tanggal pembelian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    all_states = sorted(data["customer_state"].dropna().unique())
    with st.expander("Pilih state pelanggan", expanded=False):
        selected_states = st.multiselect(
            "State", options=all_states, default=all_states, label_visibility="collapsed"
        )
    st.caption(f"{len(selected_states)} dari {len(all_states)} state dipilih")
    st.divider()
    st.caption("Sumber data: Olist E-Commerce Public Dataset (pesanan berstatus delivered).")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (data["order_purchase_timestamp"].dt.date >= start_date)
    & (data["order_purchase_timestamp"].dt.date <= end_date)
    & (data["customer_state"].isin(selected_states))
)
df = data[mask]

# ----------------------------------------------------------------------------
# Header & metrik utama
# ----------------------------------------------------------------------------
st.title("Dashboard Analisis E-Commerce Olist")
st.markdown("Performa produk, wilayah, dan pelanggan dari pesanan yang telah terkirim.")

if df.empty:
    st.warning("Tidak ada data pada filter yang dipilih. Ubah rentang tanggal atau state.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pesanan", f"{df['order_id'].nunique():,}")
c2.metric("Total Revenue (BRL)", f"{df['price'].sum():,.0f}")
c3.metric("Pelanggan Unik", f"{df['customer_unique_id'].nunique():,}")
c4.metric("Rata-rata Pengiriman", f"{df['delivery_days'].mean():.1f} hari")

st.write("")
tab_trend, tab_cat, tab_state, tab_rfm = st.tabs(
    ["Tren Bulanan", "Kategori Produk", "Wilayah & Logistik", "Segmentasi RFM"]
)

# ----------------------------------------------------------------------------
# Tab 1: Tren bulanan
# ----------------------------------------------------------------------------
with tab_trend:
    st.subheader("Tren Revenue dan Jumlah Pesanan Bulanan")
    monthly = (
        df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))
        .agg(orders=("order_id", "nunique"), revenue=("price", "sum"))
        .reset_index()
    )
    monthly["month"] = monthly["order_purchase_timestamp"].astype(str)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = new_fig()
        ax.plot(monthly["month"], monthly["revenue"], marker="o", color=TEAL, linewidth=2.5)
        ax.fill_between(monthly["month"], monthly["revenue"], color=TEAL, alpha=0.12)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_fmt))
        ax.tick_params(axis="x", rotation=60)
        finish(ax, "Revenue bulanan", ylabel="Revenue (BRL)", grid_y=True)
        show(fig)
    with col_b:
        fig, ax = new_fig()
        peak = int(monthly["orders"].idxmax())
        colors = [MUTED] * len(monthly)
        colors[peak] = TEAL
        ax.bar(monthly["month"], monthly["orders"], color=colors)
        ax.tick_params(axis="x", rotation=60)
        finish(ax, "Jumlah pesanan bulanan (puncak disorot)", ylabel="Jumlah pesanan", grid_y=True)
        show(fig)

    insight(
        f"Puncak pesanan terjadi pada **{monthly.loc[peak, 'month']}** "
        f"dengan **{monthly.loc[peak, 'orders']:,}** pesanan."
    )

# ----------------------------------------------------------------------------
# Tab 2: Kategori produk (Pertanyaan 1)
# ----------------------------------------------------------------------------
with tab_cat:
    st.subheader("Kategori Produk dengan Revenue Tertinggi dan Terendah")
    cat = (
        df.groupby("product_category_name_english")
        .agg(orders=("order_id", "nunique"), revenue=("price", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    n = st.slider("Jumlah kategori yang ditampilkan", 3, 10, 5)
    top = cat.head(n).iloc[::-1]
    bottom = cat.tail(n).iloc[::-1]

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = new_fig()
        ax.barh(top["product_category_name_english"], top["revenue"],
                color=[MUTED] * (len(top) - 1) + [TEAL])
        for i, (rev, o) in enumerate(zip(top["revenue"], top["orders"])):
            ax.text(rev, i, f"  {money_fmt(rev)} | {o:,} pesanan", va="center", fontsize=9)
        ax.set_xlim(0, top["revenue"].max() * 1.5)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(money_fmt))
        finish(ax, f"{n} kategori revenue tertinggi", xlabel="Revenue (BRL)")
        show(fig)
    with col_b:
        fig, ax = new_fig()
        ax.barh(bottom["product_category_name_english"], bottom["revenue"], color=WARN)
        for i, (rev, o) in enumerate(zip(bottom["revenue"], bottom["orders"])):
            ax.text(rev, i, f"  {rev:,.0f} | {o:,} pesanan", va="center", fontsize=9)
        ax.set_xlim(0, max(bottom["revenue"].max(), 1) * 1.6)
        finish(ax, f"{n} kategori revenue terendah", xlabel="Revenue (BRL)")
        show(fig)

    share = cat["revenue"].head(5).sum() / cat["revenue"].sum() * 100
    insight(f"5 kategori teratas menyumbang **{share:.1f}%** dari total revenue pada filter yang dipilih.")

# ----------------------------------------------------------------------------
# Tab 3: Wilayah & logistik (Pertanyaan 2 + geospasial + binning)
# ----------------------------------------------------------------------------
with tab_state:
    st.subheader("Revenue dan Waktu Pengiriman per State")
    state = (
        df.groupby("customer_state")
        .agg(orders=("order_id", "nunique"),
             revenue=("price", "sum"),
             avg_delivery_days=("delivery_days", "mean"),
             late_rate=("is_late", "mean"),
             lat=("state_lat", "first"),
             lon=("state_lng", "first"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    national_avg = df["delivery_days"].mean()
    top10 = state.head(10)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = new_fig()
        ax.bar(top10["customer_state"], top10["revenue"], color=[TEAL] + [MUTED] * (len(top10) - 1))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_fmt))
        finish(ax, "Revenue 10 state teratas", xlabel="State", ylabel="Revenue (BRL)", grid_y=True)
        show(fig)
    with col_b:
        fig, ax = new_fig()
        colors = [WARN if d > national_avg else MUTED for d in top10["avg_delivery_days"]]
        ax.bar(top10["customer_state"], top10["avg_delivery_days"], color=colors)
        ax.axhline(national_avg, color=NAVY, linestyle="--", linewidth=1)
        ax.text(len(top10) - 0.5, national_avg, f" rata-rata {national_avg:.1f} hari",
                va="bottom", ha="right", fontsize=9, color=NAVY)
        finish(ax, "Lama pengiriman (merah = di atas rata-rata)", xlabel="State", ylabel="Hari", grid_y=True)
        show(fig)

    st.subheader("Peta Sebaran Revenue per State")
    map_df = state.dropna(subset=["lat", "lon"]).copy()
    map_df["size"] = map_df["revenue"] / map_df["revenue"].max() * 400000 + 30000
    st.map(map_df, latitude="lat", longitude="lon", size="size")
    st.caption("Ukuran lingkaran menunjukkan besar revenue tiap state.")

    st.subheader("Kelompok Lama Pengiriman")
    orders_unique = df.drop_duplicates("order_id").copy()
    labels = ["Sangat cepat (<=7 hari)", "Cepat (8-14 hari)", "Normal (15-21 hari)",
              "Lambat (22-30 hari)", "Sangat lambat (>30 hari)"]
    orders_unique["delivery_group"] = pd.cut(
        orders_unique["delivery_days"], bins=[0, 7, 14, 21, 30, np.inf], labels=labels, include_lowest=True
    )
    grp = (
        orders_unique.groupby("delivery_group", observed=False)
        .agg(orders=("order_id", "count"), late_rate=("is_late", "mean"))
        .reset_index()
    )
    grp["share"] = grp["orders"] / grp["orders"].sum() * 100
    grp["late_rate"] = grp["late_rate"].fillna(0) * 100
    bin_colors = [MUTED, MUTED, MUTED, WARN, WARN]

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = new_fig(7.5, 4)
        ax.bar(grp["delivery_group"].astype(str), grp["share"], color=bin_colors)
        for i, v in enumerate(grp["share"]):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=25)
        finish(ax, "Sebaran pesanan per kelompok pengiriman", ylabel="% pesanan")
        show(fig)
    with col_b:
        fig, ax = new_fig(7.5, 4)
        ax.bar(grp["delivery_group"].astype(str), grp["late_rate"], color=bin_colors)
        for i, v in enumerate(grp["late_rate"]):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=25)
        finish(ax, "% pesanan terlambat dari estimasi", ylabel="% terlambat")
        show(fig)

    top_state = state.iloc[0]
    insight(
        f"State **{top_state['customer_state']}** menyumbang "
        f"**{top_state['revenue'] / state['revenue'].sum() * 100:.1f}%** revenue. "
        f"Rata-rata pengiriman keseluruhan **{national_avg:.1f} hari**."
    )

# ----------------------------------------------------------------------------
# Tab 4: RFM (Pertanyaan 3)
# ----------------------------------------------------------------------------
with tab_rfm:
    st.subheader("Segmentasi Pelanggan Berdasarkan RFM")
    st.caption(
        "Recency: hari sejak pembelian terakhir | Frequency: jumlah pesanan | Monetary: total belanja (BRL). "
        "Segmen dihitung ulang sesuai filter yang dipilih."
    )
    rfm = build_rfm(df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rata-rata Recency (hari)", f"{rfm['recency'].mean():.0f}")
    m2.metric("Rata-rata Frequency", f"{rfm['frequency'].mean():.2f}")
    m3.metric("Rata-rata Monetary (BRL)", f"{rfm['monetary'].mean():,.0f}")
    st.write("")

    present = [s for s in SEGMENT_ORDER if s in set(rfm["segment"])]
    seg = (
        rfm.groupby("segment")
        .agg(customers=("customer_unique_id", "count"), avg_monetary=("monetary", "mean"))
        .reindex(present)
        .reset_index()
    )
    seg["share"] = seg["customers"] / seg["customers"].sum() * 100
    seg = seg.iloc[::-1].reset_index(drop=True)
    colors = [TEAL if i == seg["customers"].idxmax() else MUTED for i in seg.index]

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = new_fig()
        ax.barh(seg["segment"], seg["customers"], color=colors)
        for i, (c, p) in enumerate(zip(seg["customers"], seg["share"])):
            ax.text(c, i, f"  {c:,} ({p:.1f}%)", va="center", fontsize=9)
        ax.set_xlim(0, seg["customers"].max() * 1.35)
        finish(ax, "Jumlah pelanggan per segmen", xlabel="Jumlah pelanggan")
        show(fig)
    with col_b:
        fig, ax = new_fig()
        ax.barh(seg["segment"], seg["avg_monetary"], color=colors)
        for i, v in enumerate(seg["avg_monetary"]):
            ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=9)
        ax.set_xlim(0, seg["avg_monetary"].max() * 1.25)
        ax.tick_params(left=False, labelleft=False)
        finish(ax, "Rata-rata monetary per segmen (BRL)", xlabel="Rata-rata total belanja (BRL)")
        show(fig)

    repeat = (rfm["frequency"] >= 2).mean() * 100
    insight(f"Hanya **{repeat:.1f}%** pelanggan yang berbelanja lebih dari satu kali pada filter yang dipilih.")

    with st.expander("Lihat 10 pelanggan dengan monetary tertinggi"):
        cols = [c for c in ["customer_unique_id", "recency", "frequency", "monetary", "segment"] if c in rfm.columns]
        st.table(rfm.sort_values("monetary", ascending=False).head(10)[cols].reset_index(drop=True))

st.caption("Proyek Analisis Data - Dicoding 2026 | Syahla Sandiani Assyifa Adam")
