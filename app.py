import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PROJECT FORESIGHT - AI DEMAND & INVENTORY DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Project Foresight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# Load external CSS
# ------------------------------------------------------------
def load_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("style.css not found. Keep style.css in the same folder as app.py.")


load_css()

# ------------------------------------------------------------
# User Data Upload
# ------------------------------------------------------------
st.sidebar.markdown("## 📂 Data Input")
st.sidebar.caption(
    "Optional: upload your own project CSV files. If all four are uploaded, they will be used by the dashboard."
)

with st.sidebar.expander("Upload Project Data", expanded=False):
    uploaded_sales = st.file_uploader(
        "Sales Daily CSV", type=["csv"], key="upload_sales"
    )
    uploaded_inventory = st.file_uploader(
        "Inventory Snapshots CSV", type=["csv"], key="upload_inventory"
    )
    uploaded_calendar = st.file_uploader(
        "Calendar CSV", type=["csv"], key="upload_calendar"
    )
    uploaded_sku = st.file_uploader("SKU Master CSV", type=["csv"], key="upload_sku")

    upload_count = sum(
        x is not None
        for x in [uploaded_sales, uploaded_inventory, uploaded_calendar, uploaded_sku]
    )
    if 0 < upload_count < 4:
        st.warning(
            "Upload all 4 CSV files to use custom data. The default project data is currently active."
        )
    elif upload_count == 4:
        st.success("✅ Custom data uploaded. Dashboard is using your CSV files.")

_upload_bytes = {
    "sales": uploaded_sales.getvalue() if uploaded_sales is not None else None,
    "inventory": (
        uploaded_inventory.getvalue() if uploaded_inventory is not None else None
    ),
    "calendar": uploaded_calendar.getvalue() if uploaded_calendar is not None else None,
    "sku": uploaded_sku.getvalue() if uploaded_sku is not None else None,
}

_use_custom_data = all(v is not None for v in _upload_bytes.values())


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
@st.cache_data
def load_data(
    sales_bytes=None, inventory_bytes=None, calendar_bytes=None, sku_bytes=None
):
    if all(
        v is not None for v in [sales_bytes, inventory_bytes, calendar_bytes, sku_bytes]
    ):
        sales = pd.read_csv(io.BytesIO(sales_bytes))
        inventory = pd.read_csv(io.BytesIO(inventory_bytes))
        calendar = pd.read_csv(io.BytesIO(calendar_bytes))
        sku_master = pd.read_csv(io.BytesIO(sku_bytes))
    else:
        sales = pd.read_csv("Dataset/sales_daily.csv")
        inventory = pd.read_csv("Dataset/inventory_snapshots.csv")
        calendar = pd.read_csv("Dataset/calendar.csv")
        sku_master = pd.read_csv("Dataset/sku_master.csv")

    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    inventory["Snapshot_Date"] = pd.to_datetime(
        inventory["Snapshot_Date"], errors="coerce"
    )
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    sku_master["Launch_Date"] = pd.to_datetime(
        sku_master["Launch_Date"], errors="coerce"
    )

    for d in [sales, inventory, sku_master]:
        d["SKU"] = d["SKU"].astype(str).str.strip()

    return sales, inventory, calendar, sku_master


@st.cache_data
def build_dataset(
    sales_bytes=None, inventory_bytes=None, calendar_bytes=None, sku_bytes=None
):
    sales, inventory, calendar, sku_master = load_data(
        sales_bytes, inventory_bytes, calendar_bytes, sku_bytes
    )

    # Sales + product master + calendar
    df = sales.merge(sku_master, on="SKU", how="left")
    df = df.merge(calendar, left_on="Date", right_on="date", how="left")
    df.drop(columns=["date"], inplace=True)

    # Monthly inventory snapshot
    df["inventory_month"] = df["Date"].dt.to_period("M")

    inventory_monthly = inventory.copy()
    inventory_monthly["inventory_month"] = inventory_monthly[
        "Snapshot_Date"
    ].dt.to_period("M")
    inventory_monthly.drop(columns=["Snapshot_Date"], inplace=True)

    # If duplicate monthly snapshots exist, keep the last one
    inventory_monthly = inventory_monthly.sort_values(
        ["SKU", "inventory_month"]
    ).drop_duplicates(["SKU", "inventory_month"], keep="last")

    df = df.merge(inventory_monthly, on=["SKU", "inventory_month"], how="left")
    df.drop(columns=["inventory_month"], inplace=True)

    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

    # Date features - same features used in the notebook
    df["day"] = df["Date"].dt.day
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["month_num"] = df["Date"].dt.month
    df["year_num"] = df["Date"].dt.year

    df["product_age_days"] = (df["Date"] - df["Launch_Date"]).dt.days.clip(lower=0)

    # Historical demand features
    g = df.groupby("SKU")["Units_Sold"]

    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)

    df["rolling_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)

    df["rolling_14"] = g.shift(1).rolling(14).mean().reset_index(level=0, drop=True)

    df["rolling_28"] = g.shift(1).rolling(28).mean().reset_index(level=0, drop=True)

    df["rolling_7_std"] = (
        g.shift(1).rolling(7).std().reset_index(level=0, drop=True).fillna(0)
    )

    # Actual future demand and targets for historical dashboard analysis
    future_cols = []
    for i in range(1, 8):
        col = f"future_{i}"
        df[col] = g.shift(-i)
        future_cols.append(col)

    df["future_7d_demand"] = df[future_cols].sum(axis=1, min_count=7)

    df["available_inventory"] = df["Current_Stock"].fillna(0) + df["On_Order"].fillna(0)

    df["stockout_risk_target"] = (
        df["available_inventory"] < df["future_7d_demand"]
    ).astype(int)

    df["reorder_qty_target"] = (
        df["future_7d_demand"]
        + df["Safety_Stock"].fillna(0)
        - df["Current_Stock"].fillna(0)
        - df["On_Order"].fillna(0)
    ).clip(lower=0)

    return df


@st.cache_resource
def load_models():
    models = {}

    model_files = {
        "demand": "Models/demand_forecast_model.pkl",
        "risk": "Models/risk_model.pkl",
        "reorder": "Models/recorder_quantity_model.pkl",
    }

    for key, filename in model_files.items():
        path = BASE_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                models[key] = pickle.load(f)

    return models


def prepare_model_input(row):
    """Create exactly the 37 feature columns used by the notebook."""
    categorical_features = [
        "SKU",
        "Category",
        "Subcategory",
        "quarter",
        "day_of_week",
        "season",
        "holiday",
        "promotion_event",
    ]

    numeric_features = [
        "Price",
        "Promotion",
        "year",
        "month",
        "week",
        "is_weekend",
        "is_holiday",
        "day",
        "day_of_year",
        "month_num",
        "year_num",
        "product_age_days",
        "Cost_Price",
        "Selling_Price",
        "Gross_Margin_Per_Unit",
        "Current_Stock",
        "On_Order",
        "Lead_Time_Days",
        "Safety_Stock",
        "Reorder_Point",
        "Inventory_Value",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_7",
        "rolling_14",
        "rolling_28",
        "rolling_7_std",
    ]

    features = categorical_features + numeric_features
    X = pd.DataFrame([row[features].to_dict()])

    for col in categorical_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def build_manual_input(
    sku,
    category,
    subcategory,
    input_date,
    launch_date,
    price,
    promotion,
    cost_price,
    selling_price,
    current_stock,
    on_order,
    lead_time_days,
    safety_stock,
    reorder_point,
    holiday,
    promotion_event,
    lag_1,
    lag_7,
    lag_14,
    lag_28,
    rolling_7,
    rolling_14,
    rolling_28,
    rolling_7_std,
):
    """Build one prediction row from user-entered business data."""
    row = {}
    input_date = pd.Timestamp(input_date)
    launch_date = pd.Timestamp(launch_date)

    month = input_date.month
    day = input_date.day
    day_of_year = input_date.dayofyear
    year = input_date.year
    week = int(input_date.isocalendar().week)
    day_of_week = input_date.day_name()
    quarter = f"Q{input_date.quarter}"
    is_weekend = int(input_date.dayofweek >= 5)
    product_age_days = max((input_date - launch_date).days, 0)

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    gross_margin = float(selling_price) - float(cost_price)
    inventory_value = float(current_stock) * float(cost_price)

    row.update(
        {
            "SKU": str(sku),
            "Category": str(category),
            "Subcategory": str(subcategory),
            "quarter": quarter,
            "day_of_week": day_of_week,
            "season": season_map[month],
            "holiday": str(holiday),
            "promotion_event": str(promotion_event),
            "Price": float(price),
            "Promotion": float(promotion),
            "year": year,
            "month": month,
            "week": week,
            "is_weekend": is_weekend,
            "is_holiday": int(str(holiday).lower() in ["1", "true", "yes"]),
            "day": day,
            "day_of_year": day_of_year,
            "month_num": month,
            "year_num": year,
            "product_age_days": product_age_days,
            "Cost_Price": float(cost_price),
            "Selling_Price": float(selling_price),
            "Gross_Margin_Per_Unit": gross_margin,
            "Current_Stock": float(current_stock),
            "On_Order": float(on_order),
            "Lead_Time_Days": float(lead_time_days),
            "Safety_Stock": float(safety_stock),
            "Reorder_Point": float(reorder_point),
            "Inventory_Value": inventory_value,
            "lag_1": float(lag_1),
            "lag_7": float(lag_7),
            "lag_14": float(lag_14),
            "lag_28": float(lag_28),
            "rolling_7": float(rolling_7),
            "rolling_14": float(rolling_14),
            "rolling_28": float(rolling_28),
            "rolling_7_std": float(rolling_7_std),
        }
    )

    return pd.Series(row)


def get_prediction_table(df, models):
    """Predict demand, stockout risk and reorder quantity for latest SKU rows."""
    latest = (
        df.sort_values(["SKU", "Date"]).groupby("SKU", as_index=False).tail(1).copy()
    )

    if latest.empty:
        return latest

    if "demand" in models:
        X = pd.concat(
            [prepare_model_input(row) for _, row in latest.iterrows()],
            ignore_index=True,
        )
        latest["Predicted_7_Day_Demand"] = np.maximum(models["demand"].predict(X), 0)

    if "risk" in models:
        X = pd.concat(
            [prepare_model_input(row) for _, row in latest.iterrows()],
            ignore_index=True,
        )
        latest["Predicted_Stockout_Risk"] = models["risk"].predict(X).astype(int)
        if hasattr(models["risk"], "predict_proba"):
            latest["Stockout_Probability"] = models["risk"].predict_proba(X)[:, 1] * 100
        else:
            latest["Stockout_Probability"] = latest["Predicted_Stockout_Risk"] * 100

    if "reorder" in models:
        X = pd.concat(
            [prepare_model_input(row) for _, row in latest.iterrows()],
            ignore_index=True,
        )
        latest["Predicted_Reorder_Qty"] = np.maximum(models["reorder"].predict(X), 0)

    return latest


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
try:
    sales, inventory, calendar, sku_master = load_data(
        _upload_bytes["sales"],
        _upload_bytes["inventory"],
        _upload_bytes["calendar"],
        _upload_bytes["sku"],
    )
    df = build_dataset(
        _upload_bytes["sales"],
        _upload_bytes["inventory"],
        _upload_bytes["calendar"],
        _upload_bytes["sku"],
    )
    models = load_models()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

prediction_df = get_prediction_table(df, models)


# ------------------------------------------------------------
# Top Navigation Tabs
# ------------------------------------------------------------
st.markdown(
    """
<div class="foresight-header">
    <div class="foresight-brand">📊 Project Foresight</div>
    <div class="foresight-subtitle">AI-powered Demand Forecasting · Inventory Intelligence · Stockout Risk Analytics</div>
    <div class="status-pill"><span class="status-dot"></span> XGBoost Intelligence System · Dashboard Online</div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🏠 Home",
        "📈 Sales Analytics",
        "🔮 Demand Forecast",
        "📦 Inventory Dashboard",
        "⚠️ Risk Dashboard",
        "🛍️ Product Details",
        "🤖 AI Prediction",
        "👔 Executive Summary",
    ]
)

with tabs[0]:
    # ============================================================
    # HOME PAGE
    # ============================================================

    st.markdown(
        '<div class="main-title">Project Foresight</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">AI-powered demand forecasting, inventory '
        "planning and stockout risk intelligence</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total SKUs", f"{df['SKU'].nunique():,}")

    with c2:
        st.metric("Sales Records", f"{len(sales):,}")

    with c3:
        st.metric("Total Units Sold", f"{sales['Units_Sold'].sum():,.0f}")

    with c4:
        st.metric("Total Revenue", f"₹{sales['Revenue'].sum():,.0f}")

    st.markdown("### Dashboard Modules")

    modules = [
        ("📈", "Sales Analytics", "Analyze sales, revenue, prices and promotions."),
        ("🔮", "Demand Forecast", "Predict 7-day future product demand."),
        (
            "📦",
            "Inventory Dashboard",
            "Monitor current stock, inventory value and reorder needs.",
        ),
        ("⚠️", "Risk Dashboard", "Identify products with stockout risk."),
        ("🛍️", "Product Details", "View product-level information and history."),
        (
            "👔",
            "Executive Summary Dashboard",
            "Management-level KPIs and business overview.",
        ),
    ]

    cols = st.columns(3)
    for i, (icon, title, description) in enumerate(modules):
        with cols[i % 3]:
            st.markdown(f"#### {icon} {title}")
            st.write(description)

    st.markdown("### Dataset Period")
    st.write(
        f"Sales data: **{sales['Date'].min().date()}** "
        f"to **{sales['Date'].max().date()}**"
    )


with tabs[1]:
    # ============================================================
    # SALES ANALYTICS
    # ============================================================
    st.title("Sales Analytics")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=sales["Date"].min().date(),
            min_value=sales["Date"].min().date(),
            max_value=sales["Date"].max().date(),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=sales["Date"].max().date(),
            min_value=sales["Date"].min().date(),
            max_value=sales["Date"].max().date(),
        )

    filtered = sales[
        (sales["Date"].dt.date >= start_date) & (sales["Date"].dt.date <= end_date)
    ].copy()

    a, b, c, d = st.columns(4)
    a.metric("Units Sold", f"{filtered['Units_Sold'].sum():,.0f}")
    b.metric("Revenue", f"₹{filtered['Revenue'].sum():,.0f}")
    c.metric("Avg. Price", f"₹{filtered['Price'].mean():,.2f}")
    d.metric("Promoted Records", f"{int(filtered['Promotion'].sum()):,}")

    daily = filtered.groupby("Date", as_index=False).agg(
        Units_Sold=("Units_Sold", "sum"), Revenue=("Revenue", "sum")
    )

    fig = px.line(daily, x="Date", y="Revenue", title="Daily Revenue")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        sku_sales = (
            filtered.groupby("SKU", as_index=False)["Units_Sold"]
            .sum()
            .sort_values("Units_Sold", ascending=False)
            .head(15)
        )
        fig = px.bar(
            sku_sales,
            x="Units_Sold",
            y="SKU",
            orientation="h",
            title="Top 15 SKUs by Units Sold",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        monthly = (
            filtered.assign(Month=filtered["Date"].dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)["Revenue"]
            .sum()
        )
        fig = px.bar(monthly, x="Month", y="Revenue", title="Monthly Revenue")
        st.plotly_chart(fig, use_container_width=True)


with tabs[2]:
    # ============================================================
    # DEMAND FORECAST
    # ============================================================
    st.title("Demand Forecast")

    if "demand" not in models:
        st.warning(
            "demand_forecast_model.pkl was not found. "
            "Place it in the same folder as app.py."
        )
        st.stop()

    sku = st.selectbox("Select SKU", sorted(df["SKU"].dropna().unique()))

    sku_df = df[df["SKU"] == sku].sort_values("Date").copy()
    latest = sku_df.iloc[-1]

    X = prepare_model_input(latest)
    predicted_demand = max(float(models["demand"].predict(X)[0]), 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected SKU", sku)
    c2.metric("Latest Daily Sales", f"{latest['Units_Sold']:.0f}")
    c3.metric("7-Day Forecast", f"{predicted_demand:.1f} units")
    c4.metric("Current Stock", f"{latest['Current_Stock']:.0f}")

    history = sku_df.tail(90)[["Date", "Units_Sold"]].rename(
        columns={"Units_Sold": "Demand"}
    )

    fig = px.line(
        history, x="Date", y="Demand", title=f"{sku} - Historical Daily Demand"
    )

    forecast_date = latest["Date"] + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(forecast_date, periods=7)

    # Show the model's 7-day total as an informational forecast line.
    # The saved model predicts the total 7-day demand, not seven separate days.
    avg_forecast = predicted_demand / 7

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=[avg_forecast] * 7,
            mode="lines+markers",
            name="Average Forecast / Day",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Forecast Inputs")

    forecast_inputs = X.T.rename(columns={0: "Value"}).copy()
    forecast_inputs["Value"] = forecast_inputs["Value"].astype(str)

    st.dataframe(
        forecast_inputs,
        use_container_width=True,
        hide_index=False,
    )


with tabs[3]:
    # ============================================================
    # INVENTORY DASHBOARD
    # ============================================================
    st.title("Inventory Dashboard")

    inv = prediction_df.copy()

    if inv.empty:
        st.warning("No inventory records available.")
        st.stop()

    total_stock = inv["Current_Stock"].fillna(0).sum()
    total_on_order = inv["On_Order"].fillna(0).sum()
    inventory_value = inv["Inventory_Value"].fillna(0).sum()

    if "Predicted_Reorder_Qty" in inv:
        total_reorder = inv["Predicted_Reorder_Qty"].sum()
    else:
        total_reorder = 0

    a, b, c, d = st.columns(4)
    a.metric("Current Stock", f"{total_stock:,.0f}")
    b.metric("On Order", f"{total_on_order:,.0f}")
    c.metric("Inventory Value", f"₹{inventory_value:,.0f}")
    d.metric("Predicted Reorder Qty", f"{total_reorder:,.0f}")

    st.markdown("### Inventory by SKU")

    display_cols = [
        "SKU",
        "Product_Name",
        "Category",
        "Current_Stock",
        "On_Order",
        "Safety_Stock",
        "Reorder_Point",
        "Inventory_Value",
    ]

    if "Predicted_Reorder_Qty" in inv:
        display_cols.append("Predicted_Reorder_Qty")

    available_cols = [c for c in display_cols if c in inv.columns]

    st.dataframe(
        inv[available_cols].sort_values("Current_Stock", ascending=True),
        use_container_width=True,
        hide_index=True,
    )

    chart = inv.nlargest(15, "Inventory_Value")[["SKU", "Inventory_Value"]]

    fig = px.bar(
        chart, x="SKU", y="Inventory_Value", title="Top 15 SKUs by Inventory Value"
    )
    st.plotly_chart(fig, use_container_width=True)


with tabs[4]:
    # ============================================================
    # RISK DASHBOARD
    # ============================================================
    st.title("Risk Dashboard")

    if "risk" not in models:
        st.warning(
            "risk_model.pkl was not found. " "Place it in the same folder as app.py."
        )
        st.stop()

    risk_df = prediction_df.copy()

    high_risk = risk_df[risk_df["Predicted_Stockout_Risk"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs Analyzed", f"{len(risk_df):,}")
    c2.metric("High-Risk SKUs", f"{len(high_risk):,}")
    c3.metric("High-Risk %", f"{(len(high_risk) / max(len(risk_df), 1)) * 100:.1f}%")

    risk_df["Risk Status"] = np.where(
        risk_df["Predicted_Stockout_Risk"] == 1, "High Risk", "Low Risk"
    )

    display_cols = [
        "SKU",
        "Product_Name",
        "Category",
        "Current_Stock",
        "On_Order",
        "Predicted_7_Day_Demand",
        "Stockout_Probability",
        "Predicted_Reorder_Qty",
        "Risk Status",
    ]
    display_cols = [c for c in display_cols if c in risk_df.columns]

    st.dataframe(
        risk_df[display_cols].sort_values("Stockout_Probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if not risk_df.empty:
        fig = px.bar(
            risk_df.nlargest(15, "Stockout_Probability"),
            x="Stockout_Probability",
            y="SKU",
            color="Risk Status",
            orientation="h",
            title="Highest Stockout Probabilities",
        )
        st.plotly_chart(fig, use_container_width=True)


with tabs[5]:
    # ============================================================
    # PRODUCT DETAILS
    # ============================================================
    st.title("Product Details")

    sku = st.selectbox(
        "Select Product / SKU", sorted(sku_master["SKU"].dropna().unique())
    )

    product = sku_master[sku_master["SKU"] == sku].iloc[0]
    history = df[df["SKU"] == sku].sort_values("Date").copy()

    st.markdown("### Product Information")

    a, b, c, d = st.columns(4)
    a.metric("Product", product["Product_Name"])
    b.metric("Category", product["Category"])
    c.metric("Subcategory", product["Subcategory"])
    d.metric("Selling Price", f"₹{product['Selling_Price']:,.2f}")

    info = pd.DataFrame(
        {
            "Field": [
                "SKU",
                "Product Name",
                "Category",
                "Subcategory",
                "Launch Date",
                "Cost Price",
                "Selling Price",
                "Gross Margin / Unit",
            ],
            "Value": [
                str(product["SKU"]),
                str(product["Product_Name"]),
                str(product["Category"]),
                str(product["Subcategory"]),
                str(product["Launch_Date"].date()),
                f"₹{float(product['Cost_Price']):,.2f}",
                f"₹{float(product['Selling_Price']):,.2f}",
                f"₹{float(product['Gross_Margin_Per_Unit']):,.2f}",
            ],
        }
    )

    info["Field"] = info["Field"].astype(str)
    info["Value"] = info["Value"].astype(str)

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            history.tail(180), x="Date", y="Units_Sold", title="Sales History"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if not prediction_df.empty:
            p = prediction_df[prediction_df["SKU"] == sku]
            if not p.empty and "Predicted_7_Day_Demand" in p:
                row = p.iloc[0]
                values = {
                    "7-Day Demand": row.get("Predicted_7_Day_Demand", 0),
                    "Current Stock": row.get("Current_Stock", 0),
                    "On Order": row.get("On_Order", 0),
                    "Reorder Qty": row.get("Predicted_Reorder_Qty", 0),
                }
                fig = px.bar(
                    x=list(values.keys()),
                    y=list(values.values()),
                    title="Current Product Planning Indicators",
                )
                st.plotly_chart(fig, use_container_width=True)


with tabs[6]:
    # ============================================================
    # AI PREDICTION - USER INPUT
    # ============================================================
    st.title("AI Prediction")
    st.caption(
        "Enter product and inventory information to generate predictions, or upload all four project CSV files from the sidebar to use your own dataset."
    )

    available_models = []
    if "demand" in models:
        available_models.append("Demand Forecast")
    if "risk" in models:
        available_models.append("Stockout Risk")
    if "reorder" in models:
        available_models.append("Reorder Quantity")

    if not available_models:
        st.error(
            "No XGBoost model files were found. Add the required .pkl files beside app.py."
        )
    else:
        st.success(
            "User input is ready. Fill in the product details below and click Predict."
        )

        use_existing = st.checkbox(
            "Use an existing SKU as a starting point", value=True
        )

        existing_sku = None
        template = None
        if use_existing and not df.empty:
            existing_sku = st.selectbox(
                "Select existing SKU", sorted(df["SKU"].dropna().astype(str).unique())
            )
            sku_history = df[df["SKU"].astype(str) == str(existing_sku)].sort_values(
                "Date"
            )
            if not sku_history.empty:
                template = sku_history.iloc[-1]

        with st.form("manual_prediction_form"):
            st.markdown("### Product Information")
            c1, c2, c3 = st.columns(3)

            with c1:
                sku_value = st.text_input(
                    "SKU",
                    value=(
                        str(template.get("SKU", "CUSTOM-001"))
                        if template is not None
                        else "CUSTOM-001"
                    ),
                )
                category_options = sorted(
                    sku_master["Category"].dropna().astype(str).unique()
                )
                default_category = (
                    str(
                        template.get(
                            "Category",
                            category_options[0] if category_options else "Unknown",
                        )
                    )
                    if template is not None
                    else (category_options[0] if category_options else "Unknown")
                )
                category = st.selectbox(
                    "Category",
                    category_options if category_options else ["Unknown"],
                    index=(
                        category_options.index(default_category)
                        if default_category in category_options
                        else 0
                    ),
                )
                sub_options = sorted(
                    sku_master["Subcategory"].dropna().astype(str).unique()
                )
                default_sub = (
                    str(
                        template.get(
                            "Subcategory", sub_options[0] if sub_options else "Unknown"
                        )
                    )
                    if template is not None
                    else (sub_options[0] if sub_options else "Unknown")
                )
                subcategory = st.selectbox(
                    "Subcategory",
                    sub_options if sub_options else ["Unknown"],
                    index=(
                        sub_options.index(default_sub)
                        if default_sub in sub_options
                        else 0
                    ),
                )

            with c2:
                input_date = st.date_input(
                    "Prediction Date",
                    value=(
                        pd.Timestamp(template["Date"]).date()
                        if template is not None
                        else pd.Timestamp.today().date()
                    ),
                )
                launch_default = (
                    pd.Timestamp(template["Launch_Date"]).date()
                    if template is not None and pd.notna(template.get("Launch_Date"))
                    else pd.Timestamp(input_date).date()
                )
                launch_date = st.date_input("Launch Date", value=launch_default)
                price = st.number_input(
                    "Price",
                    min_value=0.0,
                    value=(
                        float(template.get("Price", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
                promotion = st.number_input(
                    "Promotion",
                    min_value=0.0,
                    value=(
                        float(template.get("Promotion", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )

            with c3:
                cost_price = st.number_input(
                    "Cost Price",
                    min_value=0.0,
                    value=(
                        float(template.get("Cost_Price", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
                selling_price = st.number_input(
                    "Selling Price",
                    min_value=0.0,
                    value=(
                        float(template.get("Selling_Price", template.get("Price", 0.0)))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
                current_stock = st.number_input(
                    "Current Stock",
                    min_value=0.0,
                    value=(
                        float(template.get("Current_Stock", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
                on_order = st.number_input(
                    "On Order",
                    min_value=0.0,
                    value=(
                        float(template.get("On_Order", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )

            st.markdown("### Inventory Planning Inputs")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                lead_time = st.number_input(
                    "Lead Time (Days)",
                    min_value=0.0,
                    value=(
                        float(template.get("Lead_Time_Days", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
            with c2:
                safety_stock = st.number_input(
                    "Safety Stock",
                    min_value=0.0,
                    value=(
                        float(template.get("Safety_Stock", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
            with c3:
                reorder_point = st.number_input(
                    "Reorder Point",
                    min_value=0.0,
                    value=(
                        float(template.get("Reorder_Point", 0.0))
                        if template is not None
                        else 0.0
                    ),
                    step=1.0,
                )
            with c4:
                holiday = st.selectbox(
                    "Holiday",
                    ["0", "1"],
                    index=(
                        1
                        if template is not None
                        and str(template.get("holiday", "0")).lower()
                        in ["1", "true", "yes"]
                        else 0
                    ),
                )

            promotion_event = st.text_input(
                "Promotion Event",
                value=(
                    str(template.get("promotion_event", "None"))
                    if template is not None
                    else "None"
                ),
            )

            with st.expander("Advanced Historical Demand Inputs"):
                st.caption(
                    "These values are used by the trained model's lag and rolling-demand features. Existing SKUs are prefilled from their latest record."
                )
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    lag_1 = st.number_input(
                        "Lag 1",
                        value=(
                            float(template.get("lag_1", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                    lag_7 = st.number_input(
                        "Lag 7",
                        value=(
                            float(template.get("lag_7", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                with a2:
                    lag_14 = st.number_input(
                        "Lag 14",
                        value=(
                            float(template.get("lag_14", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                    lag_28 = st.number_input(
                        "Lag 28",
                        value=(
                            float(template.get("lag_28", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                with a3:
                    rolling_7 = st.number_input(
                        "Rolling 7",
                        value=(
                            float(template.get("rolling_7", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                    rolling_14 = st.number_input(
                        "Rolling 14",
                        value=(
                            float(template.get("rolling_14", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                with a4:
                    rolling_28 = st.number_input(
                        "Rolling 28",
                        value=(
                            float(template.get("rolling_28", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )
                    rolling_7_std = st.number_input(
                        "Rolling 7 Std",
                        min_value=0.0,
                        value=(
                            float(template.get("rolling_7_std", 0.0))
                            if template is not None
                            else 0.0
                        ),
                    )

            predict_clicked = st.form_submit_button(
                "🚀 Predict with XGBoost", use_container_width=True
            )

        if predict_clicked:
            manual_row = build_manual_input(
                sku_value,
                category,
                subcategory,
                input_date,
                launch_date,
                price,
                promotion,
                cost_price,
                selling_price,
                current_stock,
                on_order,
                lead_time,
                safety_stock,
                reorder_point,
                holiday,
                promotion_event,
                lag_1,
                lag_7,
                lag_14,
                lag_28,
                rolling_7,
                rolling_14,
                rolling_28,
                rolling_7_std,
            )
            X_manual = prepare_model_input(manual_row)

            st.markdown("### Prediction Results")
            result_cols = st.columns(3)

            if "demand" in models:
                predicted_demand = max(
                    float(models["demand"].predict(X_manual)[0]), 0.0
                )
                result_cols[0].metric(
                    "Predicted 7-Day Demand", f"{predicted_demand:,.1f} units"
                )
            else:
                predicted_demand = 0.0
                result_cols[0].warning("Demand model unavailable")

            if "risk" in models:
                risk_prediction = int(models["risk"].predict(X_manual)[0])
                if hasattr(models["risk"], "predict_proba"):
                    risk_probability = float(
                        models["risk"].predict_proba(X_manual)[0, 1] * 100
                    )
                else:
                    risk_probability = risk_prediction * 100.0
                result_cols[1].metric(
                    "Stockout Probability", f"{risk_probability:.1f}%"
                )
                if risk_prediction == 1:
                    st.error("⚠️ High stockout risk detected for this input.")
                else:
                    st.success("✅ Low stockout risk detected for this input.")
            else:
                risk_prediction = 0
                risk_probability = 0.0
                result_cols[1].warning("Risk model unavailable")

            if "reorder" in models:
                reorder_qty = max(float(models["reorder"].predict(X_manual)[0]), 0.0)
                result_cols[2].metric(
                    "Recommended Reorder Quantity", f"{reorder_qty:,.1f} units"
                )
            else:
                reorder_qty = 0.0
                result_cols[2].warning("Reorder model unavailable")

            result = pd.DataFrame(
                {
                    "Input": [
                        "SKU",
                        "Category",
                        "Current Stock",
                        "On Order",
                        "Predicted 7-Day Demand",
                        "Stockout Probability",
                        "Recommended Reorder Quantity",
                    ],
                    "Value": [
                        sku_value,
                        category,
                        current_stock,
                        on_order,
                        f"{predicted_demand:,.2f}",
                        f"{risk_probability:.2f}%",
                        f"{reorder_qty:,.2f}",
                    ],
                }
            )
            st.dataframe(result, use_container_width=True, hide_index=True)


with tabs[7]:
    # ============================================================
    # EXECUTIVE SUMMARY
    # ============================================================
    st.title("Executive Summary Dashboard")
    st.caption("Management overview of demand, sales and inventory")

    total_revenue = sales["Revenue"].sum()
    total_units = sales["Units_Sold"].sum()
    total_inventory_value = prediction_df["Inventory_Value"].fillna(0).sum()

    if "Predicted_Stockout_Risk" in prediction_df:
        risk_count = int(prediction_df["Predicted_Stockout_Risk"].sum())
    else:
        risk_count = 0

    if "Predicted_Reorder_Qty" in prediction_df:
        reorder_total = prediction_df["Predicted_Reorder_Qty"].sum()
    else:
        reorder_total = 0

    a, b, c, d, e = st.columns(5)
    a.metric("Revenue", f"₹{total_revenue:,.0f}")
    b.metric("Units Sold", f"{total_units:,.0f}")
    c.metric("Inventory Value", f"₹{total_inventory_value:,.0f}")
    d.metric("High-Risk SKUs", f"{risk_count:,}")
    e.metric("Predicted Reorder", f"{reorder_total:,.0f}")

    st.markdown("### Business Overview")

    left, right = st.columns(2)

    with left:
        monthly = (
            sales.assign(Month=sales["Date"].dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)
            .agg(Revenue=("Revenue", "sum"), Units=("Units_Sold", "sum"))
        )

        fig = px.line(
            monthly, x="Month", y="Revenue", markers=True, title="Monthly Revenue Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        category = (
            sku_master.groupby("Category", as_index=False)
            .agg(Products=("SKU", "nunique"), Avg_Price=("Selling_Price", "mean"))
            .sort_values("Products", ascending=False)
        )

        fig = px.bar(category, x="Category", y="Products", title="Products by Category")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Performance from Training Notebook")

    performance = pd.DataFrame(
        {
            "Model": [
                "XGBoost Demand Forecast",
                "XGBoost Stockout Risk",
                "XGBoost Reorder Quantity",
            ],
            "Primary Metric": [
                "R² = 94.36%",
                "Accuracy = 99.03%",
                "R² = 78.84%",
            ],
            "MAE": [
                "8.082",
                "—",
                "2.824",
            ],
            "RMSE": [
                "10.452",
                "—",
                "7.802",
            ],
        }
    )

    st.dataframe(performance, use_container_width=True, hide_index=True)

    st.info(
        "These model-performance values are the test-set results recorded "
        "in the supplied AI_Forecast notebook."
    )

st.markdown(
    '<div class="foresight-footer">Project Foresight · AI Demand & Inventory Intelligence · XGBoost Powered</div>',
    unsafe_allow_html=True,
)
