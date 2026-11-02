from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "customer_segmentation_outputs"

st.set_page_config(
    page_title="NOVA | Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ DESIGN SYSTEM ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --bg:#080e18; --panel:#101a2a; --panel2:#142238; --line:#24354c;
  --text:#edf4ff; --muted:#9aabc2; --mint:#50e3c2; --blue:#65a8ff;
  --violet:#a78bfa; --amber:#ffc46b; --rose:#ff8297;
}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 80% 0%, #152b43 0%, #080e18 42%); color:var(--text);}
.block-container {max-width:1500px; padding-top:2rem; padding-bottom:3rem;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101a2a,#0b1421); border-right:1px solid #24354c;}
[data-testid="stSidebar"] * {color:#eaf2ff;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#182338 !important;}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {background:#50e3c2;}
.hero {
  padding:2.2rem 2.4rem; border:1px solid #31506a; border-radius:25px;
  background:linear-gradient(115deg,rgba(24,54,79,.98),rgba(28,55,87,.92) 48%,rgba(31,103,105,.86));
  box-shadow:0 18px 60px rgba(0,0,0,.28); margin-bottom:1.25rem; position:relative; overflow:hidden;
}
.hero:after {content:"✦"; position:absolute; right:5%; top:-42px; font-size:190px; color:#ffffff0b;}
.eyebrow {color:#80f0d5; text-transform:uppercase; letter-spacing:.2em; font-size:.72rem; font-weight:800;}
.hero h1 {font-family:Manrope,sans-serif; color:#fff; font-weight:800; font-size:2.65rem; margin:.4rem 0 .45rem;}
.hero p {color:#d4e5f5; font-size:1rem; max-width:760px; margin:0;}
.hero-tags {margin-top:1.05rem;}
.hero-tag {display:inline-block; border:1px solid #ffffff30; background:#ffffff12; padding:.35rem .7rem; border-radius:99px; margin:0 .35rem .25rem 0; color:#e9f6ff; font-size:.75rem;}
.section-head {font-family:Manrope,sans-serif; color:#f3f7ff; font-weight:800; font-size:1.25rem; margin:.4rem 0 .2rem;}
.section-note {color:#9aabc2; font-size:.86rem; margin-bottom:.8rem;}
.kpi {background:linear-gradient(145deg,#14243a,#101a2a); border:1px solid #2a4059; border-radius:18px; padding:1rem 1.1rem; min-height:116px; box-shadow:0 10px 30px #00000020;}
.kpi-label {color:#9fb3ca; font-size:.78rem; font-weight:600; letter-spacing:.04em;}
.kpi-value {font-family:Manrope,sans-serif; color:#f5f9ff; font-size:1.8rem; font-weight:800; margin-top:.25rem;}
.kpi-foot {color:#65d9c0; font-size:.72rem; margin-top:.12rem;}
.insight {background:linear-gradient(120deg,#132a3b,#142338); border:1px solid #2c5261; border-left:4px solid #50e3c2; border-radius:14px; padding:1rem 1.1rem; color:#dceaf7; margin:.4rem 0 .8rem;}
.segment-card {background:linear-gradient(145deg,#14243a,#101a2a); border:1px solid #2a4059; border-radius:17px; padding:1rem; min-height:150px;}
.segment-name {font-weight:800; font-size:1rem; color:#f5f8ff;}
.segment-stat {font-size:1.6rem; font-weight:800; font-family:Manrope,sans-serif; margin:.35rem 0;}
.segment-desc {font-size:.78rem; color:#a8bad0;}
div[data-testid="stMetric"] {background:transparent;}
.stTabs [data-baseweb="tab-list"] {gap:8px; border-bottom:1px solid #26384e;}
.stTabs [data-baseweb="tab"] {background:#101a2a; border:1px solid #26384e; border-radius:10px 10px 0 0; color:#b9c8db; padding:.6rem 1rem;}
.stTabs [aria-selected="true"] {background:#18364b !important; color:#70e6ce !important; border-color:#3c8c83 !important;}
.stButton>button, .stDownloadButton>button {border-radius:11px; border:1px solid #3c6b79; background:#173548; color:#eaf8ff; font-weight:700;}
.stButton>button:hover, .stDownloadButton>button:hover {border-color:#50e3c2; color:#50e3c2;}
div[data-testid="stDataFrame"] {border:1px solid #26384e; border-radius:13px; overflow:hidden;}
hr {border-color:#26384e;}
.small-muted {color:#9aabc2; font-size:.78rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------ DATA LOADING ------------------------------
@st.cache_data
def load_outputs(data_dir: str):
    root = Path(data_dir)
    required = {
        "customers": root / "customer_segments.csv",
        "profiles": root / "segment_profile.csv",
        "campaigns": root / "campaign_analysis.csv",
        "summary": root / "cluster_summary.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required output files:\n" + "\n".join(missing))
    customers = pd.read_csv(required["customers"])
    profiles = pd.read_csv(required["profiles"])
    campaigns = pd.read_csv(required["campaigns"])
    summary = pd.read_csv(required["summary"])
    # Notebook exports cluster as the index in the profile and summary files.
    for frame in (profiles, campaigns, summary):
        if "Cluster" not in frame.columns:
            first = frame.columns[0]
            if str(first).lower().startswith("unnamed"):
                frame.rename(columns={first: "Cluster"}, inplace=True)
            elif first not in ("Cluster",):
                frame.rename(columns={first: "Cluster"}, inplace=True)
    for frame in (customers, profiles, campaigns, summary):
        if "Cluster" in frame.columns:
            frame["Cluster"] = pd.to_numeric(frame["Cluster"], errors="coerce")
    customers = customers.dropna(subset=["Cluster"]).copy()
    customers["Cluster"] = customers["Cluster"].astype(int)
    for frame in (profiles, campaigns, summary):
        if "Cluster" in frame.columns:
            frame.dropna(subset=["Cluster"], inplace=True)
            frame["Cluster"] = frame["Cluster"].astype(int)
    return customers, profiles, campaigns, summary

try:
    customers, profiles, campaigns, summary = load_outputs(str(DATA_DIR))
except Exception as exc:
    st.error("The dashboard data could not be loaded.")
    st.code(str(exc))
    st.info("Keep app.py beside the customer_segmentation_outputs folder. That folder must contain customer_segments.csv, segment_profile.csv, campaign_analysis.csv and cluster_summary.csv.")
    st.stop()

# ------------------------------ NORMALISE / ENRICH ------------------------------
required_cols = ["Cluster", "Income", "Total_Spending", "Total_Purchases", "Age", "Recency"]
missing_cols = [c for c in required_cols if c not in customers.columns]
if missing_cols:
    st.error(f"customer_segments.csv is missing expected columns: {', '.join(missing_cols)}")
    st.stop()

numeric_cols = customers.select_dtypes(include=np.number).columns
customers[numeric_cols] = customers[numeric_cols].replace([np.inf, -np.inf], np.nan)
customers["Cluster"] = customers["Cluster"].astype(int)

# Derive readable segment names from observed spending rank, not fixed cluster IDs.
spend_rank = customers.groupby("Cluster")["Total_Spending"].mean().sort_values()
segment_names = {}
if len(spend_rank) == 3:
    segment_names = {
        spend_rank.index[0]: "Low-Engagement",
        spend_rank.index[1]: "Regular Mid-Value",
        spend_rank.index[2]: "Premium High-Value",
    }
else:
    labels = ["Emerging Value", "Developing Value", "Core Customers", "High Value",
              "Elite Customers", "Strategic Customers", "Top Tier", "VIP Customers"]
    segment_names = {cluster: labels[min(i, len(labels)-1)] for i, cluster in enumerate(spend_rank.index)}
customers["Segment"] = customers["Cluster"].map(segment_names).fillna(customers["Cluster"].astype(str))
profiles["Segment"] = profiles["Cluster"].map(segment_names).fillna(profiles["Cluster"].astype(str))
campaigns["Segment"] = campaigns["Cluster"].map(segment_names).fillna(campaigns["Cluster"].astype(str))
summary["Segment"] = summary["Cluster"].map(segment_names).fillna(summary["Cluster"].astype(str))

# Friendly formatting helpers
money_cols = ["Income", "Total_Spending"]
def money(v):
    return f"₹{v:,.0f}" if pd.notna(v) else "—"

# ------------------------------ SIDEBAR FILTERS ------------------------------
with st.sidebar:
    st.markdown("## ✦ NOVA / CONTROL ROOM")
    st.caption("CUSTOMER INTELLIGENCE")
    st.divider()
    all_segments = list(segment_names.values())
    chosen_segments = st.multiselect("Customer segments", all_segments, default=all_segments)
    min_age = int(np.nanmin(customers["Age"])) if customers["Age"].notna().any() else 18
    max_age = int(np.nanmax(customers["Age"])) if customers["Age"].notna().any() else 100
    age_range = st.slider("Age range", min_value=min_age, max_value=max(max_age, min_age+1),
                          value=(min_age, max_age))
    income_min = int(max(0, customers["Income"].min(skipna=True)))
    income_max = int(customers["Income"].max(skipna=True))
    income_range = st.slider("Annual income", min_value=income_min, max_value=max(income_max, income_min+1),
                             value=(income_min, income_max), step=max(1, (income_max-income_min)//100))
    st.divider()
    st.markdown("### Explore behaviour")
    channel = st.selectbox("Preferred purchase channel", ["All channels", "Web", "Catalog", "Store"])
    st.markdown("<div class='small-muted'>Filters update every chart, profile and customer record.</div>", unsafe_allow_html=True)

filtered = customers[
    customers["Segment"].isin(chosen_segments)
    & customers["Age"].between(age_range[0], age_range[1], inclusive="both")
    & customers["Income"].between(income_range[0], income_range[1], inclusive="both")
].copy()
if channel != "All channels":
    channel_col = {"Web": "NumWebPurchases", "Catalog": "NumCatalogPurchases", "Store": "NumStorePurchases"}[channel]
    if channel_col in filtered.columns:
        filtered = filtered[filtered[channel_col] > 0]

# ------------------------------ HEADER ------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ Customer intelligence suite / 2026</div>
  <h1>Know your customers.<br>Grow with precision.</h1>
  <p>A unified view of customer value, purchase behaviour and campaign engagement — turning six clustering approaches into practical business decisions.</p>
  <div class="hero-tags">
    <span class="hero-tag">K-Means final segmentation</span>
    <span class="hero-tag">6 algorithms evaluated</span>
    <span class="hero-tag">Behaviour-led insights</span>
    <span class="hero-tag">Interactive exploration</span>
  </div>
</div>
""", unsafe_allow_html=True)

total_customers = len(filtered)
avg_income = filtered["Income"].mean() if total_customers else np.nan
avg_spend = filtered["Total_Spending"].mean() if total_customers else np.nan
response_rate = filtered["Response"].mean()*100 if total_customers and "Response" in filtered else np.nan
avg_purchase = filtered["Total_Purchases"].mean() if total_customers else np.nan

kpis = st.columns(4)
kpi_data = [
    ("CUSTOMERS IN VIEW", f"{total_customers:,}", "Filtered customer records"),
    ("AVERAGE ANNUAL INCOME", money(avg_income), "Mean household income"),
    ("AVERAGE CUSTOMER SPEND", money(avg_spend), "Across six product categories"),
    ("CAMPAIGN RESPONSE", f"{response_rate:.1f}%" if pd.notna(response_rate) else "—", "Last campaign response"),
]
for col, (label, value, foot) in zip(kpis, kpi_data):
    col.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-foot'>{foot}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
tabs = st.tabs(["✦ Executive overview", "◈ Segment intelligence", "↗ Campaign studio", "⌕ Customer explorer", "◎ Model & methodology", "✧ Growth playbook"])

# Consistent plot styling
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#dce8f7"), margin=dict(l=20,r=20,t=55,b=25))
PALETTE = ["#50e3c2", "#65a8ff", "#ff8297", "#ffc46b", "#a78bfa", "#78d6f0"]

with tabs[0]:
    st.markdown("<div class='section-head'>The customer landscape</div><div class='section-note'>A high-level view of who your customers are and how they contribute.</div>", unsafe_allow_html=True)
    if filtered.empty:
        st.warning("No customers match the selected filters. Widen the filters in the sidebar.")
    else:
        left, right = st.columns([1, 1.2])
        with left:
            counts = filtered.groupby("Segment").size().reset_index(name="Customers")
            fig = px.pie(counts, names="Segment", values="Customers", hole=.68,
                         color_discrete_sequence=PALETTE, title="Segment mix")
            fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#0b1421", width=3)))
            fig.update_layout(**PLOT, legend=dict(orientation="h", y=-.12))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            spend_by = filtered.groupby("Segment", as_index=False).agg(
                Average_Spending=("Total_Spending","mean"),
                Average_Income=("Income","mean"),
                Customers=("ID","count") if "ID" in filtered else ("Cluster","count"))
            fig = px.bar(spend_by, x="Segment", y="Average_Spending", color="Segment",
                         text_auto=".0f", color_discrete_sequence=PALETTE, title="Average spend by segment")
            fig.update_layout(**PLOT, showlegend=False, xaxis_title="", yaxis_title="Average spend")
            st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns(2)
        with left:
            fig = px.scatter(filtered, x="Income", y="Total_Spending", color="Segment",
                             size="Total_Purchases", hover_data=["ID","Age","Recency"] if "ID" in filtered else ["Age","Recency"],
                             color_discrete_sequence=PALETTE, title="Income vs. spending power", opacity=.78)
            fig.update_layout(**PLOT, xaxis_title="Annual income", yaxis_title="Total spend")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            channels = [c for c in ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"] if c in filtered]
            if channels:
                channel_data = filtered.groupby("Segment")[channels].mean().reset_index().melt(
                    id_vars="Segment", var_name="Channel", value_name="Average purchases")
                channel_data["Channel"] = channel_data["Channel"].map({
                    "NumWebPurchases":"Web", "NumCatalogPurchases":"Catalog", "NumStorePurchases":"Store"})
                fig = px.bar(channel_data, x="Segment", y="Average purchases", color="Channel",
                             barmode="group", color_discrete_sequence=PALETTE, title="How customers prefer to buy")
                fig.update_layout(**PLOT, xaxis_title="", yaxis_title="Average purchases")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='section-head'>What stands out</div>", unsafe_allow_html=True)
        top_segment = filtered.groupby("Segment")["Total_Spending"].mean().idxmax()
        top_share = filtered["Segment"].value_counts(normalize=True).max()*100
        st.markdown(f"<div class='insight'><b>Highest-value segment:</b> {top_segment} currently has the highest average spend. Use the segment intelligence and campaign tabs to tailor retention and acquisition actions. <br><br><b>Largest segment share:</b> {top_share:.1f}% of the filtered audience belongs to the largest segment.</div>", unsafe_allow_html=True)
        # Dynamic signals calculated from the currently filtered customer records.
        spend_by_segment = filtered.groupby("Segment")["Total_Spending"].mean().sort_values(ascending=False)
        recency_by_segment = filtered.groupby("Segment")["Recency"].mean()
        purchase_by_segment = filtered.groupby("Segment")["Total_Purchases"].mean()
        sig1, sig2, sig3 = st.columns(3)
        sig1.metric("Highest average spend", str(spend_by_segment.index[0]), money(spend_by_segment.iloc[0]))
        sig2.metric("Most recent activity", str(recency_by_segment.idxmin()), f"{recency_by_segment.min():.0f} days average recency")
        sig3.metric("Most frequent purchasers", str(purchase_by_segment.idxmax()), f"{purchase_by_segment.max():.1f} average purchases")
        st.caption("Signals recalculate with the sidebar filters. They describe observed averages, not causal effects or forecasts.")

with tabs[1]:
    st.markdown("<div class='section-head'>Understand the people behind each cluster</div><div class='section-note'>Segment names are assigned by observed average spending rank; cluster IDs remain available for traceability.</div>", unsafe_allow_html=True)
    if not filtered.empty:
        prof = filtered.groupby("Segment").agg(
            Customers=("Cluster","size"), Avg_Age=("Age","mean"), Avg_Income=("Income","mean"),
            Avg_Spend=("Total_Spending","mean"), Avg_Purchases=("Total_Purchases","mean"),
            Avg_Recency=("Recency","mean"), Avg_Deals=("NumDealsPurchases","mean") if "NumDealsPurchases" in filtered else ("Cluster","mean")
        ).round(1).reset_index()
        prof["Customer Share"] = (prof["Customers"]/len(filtered)*100).round(1)
        cards = st.columns(min(3, max(1, len(prof))))
        for i, row in prof.iterrows():
            with cards[i % len(cards)]:
                st.markdown(f"""<div class="segment-card">
                <div class="segment-name">{row['Segment']}</div>
                <div class="segment-stat">{int(row['Customers']):,} <span style="font-size:.8rem;color:#9aabc2">customers</span></div>
                <div class="segment-desc">{row['Customer Share']:.1f}% of filtered audience · Avg spend {money(row['Avg_Spend'])}<br>Avg income {money(row['Avg_Income'])} · Avg age {row['Avg_Age']:.0f}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("#### Segment comparison")
        st.dataframe(prof.rename(columns={
            "Avg_Age":"Average age","Avg_Income":"Average income","Avg_Spend":"Average spend",
            "Avg_Purchases":"Average purchases","Avg_Recency":"Days since last purchase",
            "Avg_Deals":"Average deal purchases"
        }), use_container_width=True, hide_index=True)
        heat_cols = [c for c in ["Age","Income","Recency","Total_Spending","Total_Purchases",
                                 "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases"] if c in filtered]
        heat = filtered.groupby("Segment")[heat_cols].mean()
        # Column-wise standardisation makes differently scaled metrics comparable in the heatmap.
        heat_z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
        fig = px.imshow(heat_z, aspect="auto", color_continuous_scale="Tealrose",
                        title="Relative segment behaviour (standardised averages)")
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Heatmap values show relative differences from the overall segment average, not raw currency or counts.")
    else:
        st.info("No customers match the current filters.")

with tabs[2]:
    st.markdown("<div class='section-head'>Campaign performance, by audience</div><div class='section-note'>Campaign acceptance is evaluated after clustering; campaign outcomes are not used to define the customer segments.</div>", unsafe_allow_html=True)
    if not filtered.empty and "Response" in filtered:
        campaign_cols = [c for c in ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"] if c in filtered]
        camp = filtered.groupby("Segment")[campaign_cols].mean().mul(100).reset_index().melt(
            id_vars="Segment", var_name="Campaign", value_name="Acceptance rate (%)")
        camp["Campaign"] = camp["Campaign"].replace({
            "AcceptedCmp1":"Campaign 1","AcceptedCmp2":"Campaign 2","AcceptedCmp3":"Campaign 3",
            "AcceptedCmp4":"Campaign 4","AcceptedCmp5":"Campaign 5","Response":"Last campaign"})
        fig = px.bar(camp, x="Campaign", y="Acceptance rate (%)", color="Segment",
                     barmode="group", color_discrete_sequence=PALETTE, title="Acceptance rate by campaign and segment")
        fig.update_layout(**PLOT, yaxis_title="Acceptance rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        response = filtered.groupby("Segment")["Response"].mean().mul(100).sort_values(ascending=False).reset_index(name="Response rate")
        left, right = st.columns([1,1])
        with left:
            fig = px.bar(response, x="Segment", y="Response rate", color="Segment",
                         text_auto=".1f", color_discrete_sequence=PALETTE, title="Last campaign response rate")
            fig.update_layout(**PLOT, showlegend=False, yaxis_title="Response rate (%)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            best = response.iloc[0]
            st.markdown(f"""<div class="insight" style="margin-top:2rem">
            <div class="eyebrow">Campaign opportunity</div>
            <h3 style="color:#fff;margin:.4rem 0">{best['Segment']}</h3>
            <div style="font-size:2rem;font-weight:800;color:#50e3c2">{best['Response rate']:.1f}%</div>
            <p style="color:#c5d6e8">Highest observed last-campaign response rate in the current filtered audience. Prioritise testing tailored offers with this group.</p>
            </div>""", unsafe_allow_html=True)
        st.dataframe(camp.pivot(index="Segment", columns="Campaign", values="Acceptance rate (%)").round(2),
                     use_container_width=True)
    else:
        st.info("Campaign response data is unavailable for the current selection.")

with tabs[3]:
    st.markdown("<div class='section-head'>Find and inspect individual customers</div><div class='section-note'>Search by customer ID, narrow the list and export the filtered audience for follow-up.</div>", unsafe_allow_html=True)
    if not filtered.empty:
        c1,c2,c3 = st.columns([1.2,1,1])
        with c1: query = st.text_input("Search customer ID", placeholder="e.g. 8720")
        with c2: segment_choice = st.selectbox("Segment", ["All"] + sorted(filtered["Segment"].unique().tolist()))
        with c3: sort_choice = st.selectbox("Sort customers by", ["Highest spend","Highest income","Most recent purchase","Most purchases"])
        view = filtered.copy()
        if query.strip() and "ID" in view:
            view = view[view["ID"].astype(str).str.contains(query.strip(), case=False, na=False)]
        if segment_choice != "All":
            view = view[view["Segment"] == segment_choice]
        sort_map = {"Highest spend":("Total_Spending",False),"Highest income":("Income",False),
                    "Most recent purchase":("Recency",True),"Most purchases":("Total_Purchases",False)}
        sort_col, asc = sort_map[sort_choice]
        view = view.sort_values(sort_col, ascending=asc)
        show_cols = [c for c in ["ID","Segment","Cluster","Age","Education","Marital_Status","Income",
                                  "Total_Spending","Total_Purchases","Recency","Response"] if c in view]
        st.caption(f"{len(view):,} customer records match your search.")
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
        st.download_button("⬇ Download filtered customers", view.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_customer_segments.csv", mime="text/csv")
        if len(view):
            chosen_id = st.selectbox("Open customer snapshot", view["ID"].tolist(), format_func=lambda x: f"Customer {x}") if "ID" in view else None
            if chosen_id is not None:
                record = view[view["ID"] == chosen_id].iloc[0]
                a,b,c,d = st.columns(4)
                a.metric("Segment", str(record["Segment"]))
                b.metric("Annual income", money(record["Income"]))
                c.metric("Total spend", money(record["Total_Spending"]))
                d.metric("Days since purchase", f"{record['Recency']:.0f}")
    else:
        st.info("No customers match the current filters.")

with tabs[4]:
    st.markdown("<div class='section-head'>Model transparency</div><div class='section-note'>The dashboard presents the final K-Means segmentation from the analysis notebook and its post-clustering profiles.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight">
    <b>Segmentation approach</b><br>
    Customer behaviour variables are preprocessed using median imputation, robust scaling for numeric variables and one-hot encoding for categorical variables. K-Means is used for the final three-segment customer view shown in this dashboard.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### Dataset coverage")
    d1,d2,d3 = st.columns(3)
    d1.metric("Customer records", f"{len(customers):,}")
    d2.metric("Final segments", f"{customers['Cluster'].nunique():,}")
    d3.metric("Behavioural features", "20+")
    st.markdown("#### Segment population")
    if "Customer_Count" in summary.columns:
        st.dataframe(summary[["Cluster","Segment","Customer_Count"] + (["Percentage"] if "Percentage" in summary else [])],
                     use_container_width=True, hide_index=True)
    st.markdown("#### Interpretation guide")
    st.write("- **Silhouette score:** higher values generally indicate better-separated clusters.")
    st.write("- **Davies–Bouldin index:** lower values generally indicate more compact, separated clusters.")
    st.write("- **Calinski–Harabasz index:** higher values generally indicate stronger between-cluster separation.")
    st.caption("The notebook compares K-Means, Agglomerative, Gaussian Mixture, DBSCAN, BIRCH and OPTICS. This dashboard focuses on the selected K-Means segmentation; refer to the notebook's model-evaluation table for the complete algorithm comparison.")


with tabs[5]:
    st.markdown("<div class='section-head'>Turn customer understanding into action</div><div class='section-note'>A practical playbook generated from observed segment characteristics. Recommendations are hypotheses to test, not guaranteed outcomes.</div>", unsafe_allow_html=True)
    if filtered.empty:
        st.info("No customers match the current filters. Widen the sidebar filters to generate a playbook.")
    else:
        agg_dict = {
            "Customers": ("Cluster", "size"),
            "Spend": ("Total_Spending", "mean"),
            "Income": ("Income", "mean"),
            "Recency": ("Recency", "mean"),
            "Purchases": ("Total_Purchases", "mean"),
        }
        if "Response" in filtered.columns:
            agg_dict["Response"] = ("Response", "mean")
        else:
            agg_dict["Response"] = ("Cluster", "mean")
        playbook = filtered.groupby("Segment").agg(**agg_dict).reset_index()
        spend_mid = playbook["Spend"].median()
        recency_mid = playbook["Recency"].median()
        response_mid = playbook["Response"].median()
        for _, r in playbook.sort_values("Spend", ascending=False).iterrows():
            if r["Spend"] >= spend_mid and r["Recency"] <= recency_mid:
                priority = "RETENTION"
                action = "Protect loyalty with early access, personalised bundles and service-led benefits. Test a loyalty message before offering a discount."
            elif r["Recency"] > recency_mid:
                priority = "RE-ENGAGEMENT"
                action = "Run a win-back test with a relevant product reminder and a time-limited incentive. Compare it with a no-discount reminder group."
            elif r["Response"] >= response_mid:
                priority = "CAMPAIGN EXPANSION"
                action = "Test adjacent product recommendations and a tailored campaign. Keep a holdout group to estimate incremental lift."
            else:
                priority = "DEVELOP VALUE"
                action = "Test cross-sell bundles, product education and low-friction offers to encourage repeat purchases or category exploration."
            cluster_id = next((k for k, v in segment_names.items() if v == r["Segment"]), "—")
            with st.container(border=True):
                left, right = st.columns([1, 2.2])
                with left:
                    st.markdown(f"**{r['Segment']}**")
                    st.caption(f"Cluster {cluster_id}")
                    st.metric("Customers", f"{int(r['Customers']):,}")
                    st.metric("Average spend", money(r["Spend"]))
                with right:
                    st.markdown(f"**{priority}**")
                    st.write(action)
                    st.caption(f"Observed averages: income {money(r['Income'])}, recency {r['Recency']:.1f} days, purchases {r['Purchases']:.1f}, last-campaign response {r['Response']*100:.1f}%.")
        st.markdown("#### Segment-level action matrix")
        matrix = playbook[["Segment", "Customers", "Spend", "Recency", "Purchases", "Response"]].copy()
        matrix["Response"] = (matrix["Response"] * 100).round(1)
        matrix = matrix.rename(columns={
            "Customers": "Customer count", "Spend": "Average spend",
            "Recency": "Average recency (days)", "Purchases": "Average purchases",
            "Response": "Last campaign response (%)"
        })
        st.dataframe(matrix, use_container_width=True, hide_index=True)
        st.download_button(
            "Download segment growth playbook",
            matrix.to_csv(index=False).encode("utf-8"),
            file_name="nova_segment_growth_playbook.csv",
            mime="text/csv"
        )
        st.markdown("#### Recommended measurement plan")
        st.write("- Define one primary outcome before launching a campaign, such as conversion or repeat purchase.")
        st.write("- Keep a holdout group so campaign impact can be separated from normal customer behaviour.")
        st.write("- Compare results by segment and channel; do not claim incremental lift without a suitable comparison.")
        st.write("- Revisit segment profiles periodically because customer behaviour can change over time.")

st.markdown("---")
st.markdown("<div class='small-muted'>NOVA CUSTOMER INTELLIGENCE · Built with Streamlit & Plotly · Segment labels describe observed spending behaviour; campaign recommendations are hypotheses to test.</div>", unsafe_allow_html=True)
