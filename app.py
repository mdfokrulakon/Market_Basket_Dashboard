import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Retail Insights | FP-Growth Market Basket Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. COLOR SYSTEM & GLOBAL PLOTLY THEME
# ==============================================================================
# A single palette used everywhere -> charts, cards, badges all feel like one product.
COLORS = {
    "primary":   "#4F46E5",  # indigo
    "primary_d": "#3730A3",
    "violet":    "#7C3AED",
    "pink":      "#DB2777",
    "blue":      "#0284C7",
    "teal":      "#0D9488",
    "green":     "#16A34A",
    "amber":     "#D97706",
    "orange":    "#EA580C",
    "red":       "#DC2626",
    "slate":     "#334155",
    "bg":        "#F5F6FB",
}

CATEGORICAL_SEQUENCE = [
    COLORS["primary"], COLORS["pink"], COLORS["teal"], COLORS["amber"],
    COLORS["blue"], COLORS["violet"], COLORS["green"], COLORS["orange"],
]

pio.templates["retail_theme"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, -apple-system, sans-serif", size=13, color=COLORS["slate"]),
        colorway=CATEGORICAL_SEQUENCE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=17, color="#0F172A", family="Inter, sans-serif")),
        xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB", linecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB", linecolor="#CBD5E1"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, l=10, r=10, b=10),
    )
)
pio.templates.default = "retail_theme"

BLUE_PURPLE_SCALE = ["#E0E7FF", "#A5B4FC", "#818CF8", "#6366F1", "#4F46E5", "#4338CA", "#3730A3"]
VIRIDIS_LIKE_SCALE = ["#0D9488", "#0284C7", "#4F46E5", "#7C3AED", "#DB2777"]

# ==============================================================================
# 3. CUSTOM CSS
# ==============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .main {{
        background-color: {COLORS['bg']};
    }}

    /* ---------- Hero Header ---------- */
    .hero-banner {{
        background: linear-gradient(120deg, {COLORS['primary']} 0%, {COLORS['violet']} 55%, {COLORS['pink']} 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }}
    .hero-banner h1 {{
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }}
    .hero-banner p {{
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
        margin: 0;
        font-weight: 500;
    }}

    /* ---------- KPI Metric Cards ---------- */
    .metric-card {{
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
        border: 1px solid #EEF0F6;
        border-left: 5px solid var(--accent, {COLORS['primary']});
        margin-bottom: 15px;
        transition: transform 0.15s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
    }}
    .metric-card .metric-icon {{
        font-size: 1.3rem;
        margin-bottom: 6px;
        display: inline-block;
    }}
    .metric-card h4 {{
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 6px 0;
    }}
    .metric-card h2 {{
        color: #0F172A;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .metric-card .metric-sub {{
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 4px;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #E0E7FF !important;
    }}
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {{
        color: #C7D2FE !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}
    section[data-testid="stSidebar"] .stExpander {{
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }}
    div[data-baseweb="slider"] > div > div {{
        background: #6366F1 !important;
    }}

    /* ---------- Insight Box ---------- */
    .insight-box {{
        background: linear-gradient(135deg, #EEF2FF 0%, #FAF5FF 100%);
        border-left: 5px solid {COLORS['primary']};
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0px;
        color: #312E81;
        box-shadow: 0 2px 10px rgba(79,70,229,0.08);
        line-height: 1.7;
    }}
    .insight-box b {{ color: {COLORS['primary_d']}; }}

    .warn-box {{
        background: linear-gradient(135deg, #FEF3C7 0%, #FFF7ED 100%);
        border-left: 5px solid {COLORS['amber']};
        border-radius: 10px;
        padding: 16px 20px;
        color: #92400E;
        font-weight: 500;
    }}

    .info-box {{
        background: linear-gradient(135deg, #E0F2FE 0%, #EFF6FF 100%);
        border-left: 5px solid {COLORS['blue']};
        border-radius: 10px;
        padding: 16px 20px;
        color: #075985;
        font-weight: 500;
    }}

    /* ---------- Section Headings ---------- */
    .section-header {{
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }}
    .section-sub {{
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 14px;
        font-weight: 500;
    }}

    /* ---------- Recommendation Cards ---------- */
    .rec-card {{
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid #EEF0F6;
        border-left: 5px solid {COLORS['green']};
        box-shadow: 0 3px 12px rgba(15,23,42,0.05);
    }}
    .rec-card .rec-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
    }}
    .badge-conf {{ background: #DBEAFE; color: #1D4ED8; }}
    .badge-lift {{ background: #DCFCE7; color: #15803D; }}
    .badge-reason {{ background: #F3E8FF; color: #7E22CE; }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #ffffff;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: #64748B;
        border: 1px solid #EEF0F6;
        border-bottom: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(120deg, {COLORS['primary']}, {COLORS['violet']}) !important;
        color: #ffffff !important;
    }}

    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #EEF0F6;
    }}

    /* ---------- Progress-bar cell look ---------- */
    .metric-pill-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }}
</style>
""", unsafe_allow_html=True)


def metric_card_html(icon, label, value, accent, sub=""):
    return f"""
    <div class="metric-card" style="--accent:{accent};">
        <div class="metric-icon">{icon}</div>
        <h4>{label}</h4>
        <h2>{value}</h2>
        <div class="metric-sub">{sub}</div>
    </div>
    """


# ==============================================================================
# 4. HIGH-PERFORMANCE DATA PIPELINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_path="groceries - groceries.csv"):
    raw_df = pd.read_csv(file_path)
    item_cols = [c for c in raw_df.columns if c.lower().startswith('item ')]

    if len(item_cols) > 1:
        def extract_basket(row):
            items = [str(x).strip() for x in row[item_cols] if pd.notna(x) and str(x).strip() != '']
            return list(dict.fromkeys(items))

        transactions = raw_df.apply(extract_basket, axis=1).tolist()
    else:
        cols = raw_df.columns
        tx_col = cols[0]
        item_col = cols[1]
        raw_df[item_col] = raw_df[item_col].astype(str).str.strip()
        transactions = raw_df[raw_df[item_col] != ''].groupby(tx_col)[item_col].apply(lambda x: list(dict.fromkeys(x))).tolist()

    transactions = [t for t in transactions if len(t) > 0]
    all_products = sorted(list(set(item for t in transactions for item in t)))

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    return encoded_df, len(transactions), len(all_products), all_products


try:
    encoded_df, total_tx, total_products, all_products = load_and_clean_data()
except Exception as e:
    st.markdown(f'<div class="warn-box">⚠️ Error loading dataset file: {e}</div>', unsafe_allow_html=True)
    st.stop()

# ==============================================================================
# 5. SIDEBAR: SCROLLABLE PARAMETERS & EXPLANATORY GUIDES
# ==============================================================================
st.sidebar.title("🎛️ Control Panel")
st.sidebar.caption("Tune hyperparameters in real time")

with st.sidebar.container():
    st.markdown('<div class="param-scroll-container">', unsafe_allow_html=True)

    st.subheader("1. FP-Growth Thresholds")

    min_support = st.slider(
        "Minimum Support (minSupport)",
        min_value=0.005, max_value=0.100, value=0.020, step=0.005, format="%.3f",
        help="Filters itemsets appearing in at least this fraction of all transactions."
    )

    min_confidence = st.slider(
        "Minimum Confidence (minConfidence)",
        min_value=0.05, max_value=0.80, value=0.20, step=0.05, format="%.2f",
        help="Filters rules where probability of buying Consequent given Antecedent meets this threshold."
    )

    st.subheader("2. Result Controls")

    metric_sort = st.selectbox(
        "Sort Association Rules By",
        options=["lift", "confidence", "support"],
        index=0,
        format_func=lambda x: x.capitalize()
    )

    top_n = st.slider("Display Limit (Top N)", min_value=5, max_value=50, value=15)

    st.markdown("---")
    st.subheader("📖 Parameter Guide & Concepts")

    with st.expander("💡 What is Support?"):
        st.write(r"""
        **Support** measures how frequently an item or itemset appears in the dataset.

        $$\text{Support}(A) = \frac{\text{Transactions with } A}{\text{Total Transactions}}$$
        """)

    with st.expander("🎯 What is Confidence?"):
        st.write(r"""
        **Confidence** measures the conditional probability that product **B** is bought when product **A** is bought.

        $$\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}$$
        """)

    with st.expander("🚀 What is Lift?"):
        st.write(r"""
        **Lift** quantifies rule strength relative to random chance.
        * **Lift > 1**: Positive purchase correlation (Best rules).
        * **Lift = 1**: Independent items.
        * **Lift < 1**: Negative correlation (Substitutes).

        $$\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$$
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 6. DASHBOARD HEADER & KPI CARDS
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <h1>🛍️ Retail Intelligence: Market Basket Analytics</h1>
    <p>Powered by FP-Growth & MLxtend Engine · Enterprise-Grade Cross-Sell Insights</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(metric_card_html("🧺", "Total Baskets", f"{total_tx:,}", COLORS["primary"], "Transactions analyzed"), unsafe_allow_html=True)

with col2:
    st.markdown(metric_card_html("📦", "Unique Products", f"{total_products:,}", COLORS["blue"], "SKUs in catalog"), unsafe_allow_html=True)

# Model Execution
freq_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)

if not freq_itemsets.empty:
    freq_itemsets['items'] = freq_itemsets['itemsets'].apply(lambda x: ", ".join(list(x)))
    freq_itemsets['item_count'] = freq_itemsets['itemsets'].apply(lambda x: len(x))
    freq_itemsets['freq'] = (freq_itemsets['support'] * total_tx).round().astype(int)
    freq_itemsets['support_pct'] = (freq_itemsets['support'] * 100).round(2)
    freq_itemsets = freq_itemsets.sort_values(by='freq', ascending=False)

    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ", ".join(list(x)))
        rules['consequent_str'] = rules['consequents'].apply(lambda x: ", ".join(list(x)))
        rules['rule_str'] = rules['antecedent_str'] + " ➔ " + rules['consequent_str']
        rules = rules.sort_values(by=metric_sort, ascending=False)
else:
    rules = pd.DataFrame()

with col3:
    st.markdown(metric_card_html("🔥", "Frequent Patterns", f"{len(freq_itemsets):,}", COLORS["teal"], f"At ≥{min_support:.1%} support"), unsafe_allow_html=True)

with col4:
    st.markdown(metric_card_html("🔗", "Association Rules", f"{len(rules):,}", COLORS["pink"], f"At ≥{min_confidence:.0%} confidence"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 7. DASHBOARD MAIN TABBED INTERFACE
# ==============================================================================
tab_overview, tab_itemsets, tab_rules, tab_recommender = st.tabs([
    "📊 Executive Summary",
    "🔥 Frequent Itemsets",
    "🔗 Association Rules & Graphs",
    "🔮 Real-Time Cross-Seller"
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW & EXPLANATIONS
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-header">💡 Automated Executive Briefing</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Auto-generated insight from the strongest rule under current thresholds</div>', unsafe_allow_html=True)

    if not rules.empty:
        max_lift_rule = rules.sort_values(by='lift', ascending=False).iloc[0]

        st.markdown(f"""
        <div class="insight-box">
            <b>Key Business Takeaway:</b><br>
            • The highest affinity rule discovered is <b>{max_lift_rule['antecedent_str']} ➔ {max_lift_rule['consequent_str']}</b>
            with a <b>Lift of {max_lift_rule['lift']:.2f}x</b>.<br>
            • Customers who purchase <i>{max_lift_rule['antecedent_str']}</i> are <b>{max_lift_rule['lift']:.2f} times more likely</b>
            to buy <i>{max_lift_rule['consequent_str']}</i> than an average customer.<br>
            • This rule demonstrates a <b>Confidence of {max_lift_rule['confidence']*100:.1f}%</b> across {int(max_lift_rule['support']*total_tx)} historical transactions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">No rules detected. Lower <b>minSupport</b> or <b>minConfidence</b> in the sidebar to generate insights.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown('<div class="section-header" style="font-size:1.05rem;">Top 10 Individual Product Frequencies</div>', unsafe_allow_html=True)
        single_items = freq_itemsets[freq_itemsets['item_count'] == 1].head(10)

        fig1 = px.bar(
            single_items,
            x='support_pct',
            y='items',
            orientation='h',
            text='support_pct',
            color='support_pct',
            color_continuous_scale=BLUE_PURPLE_SCALE,
            labels={'support_pct': 'Support (% of Total Baskets)', 'items': 'Product'}
        )
        fig1.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=400, coloraxis_showscale=False)
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.markdown('<div class="section-header" style="font-size:1.05rem;">Rule Lift vs. Confidence Distribution</div>', unsafe_allow_html=True)
        if not rules.empty:
            fig2 = px.scatter(
                rules,
                x='confidence',
                y='lift',
                size='support',
                color='lift',
                hover_data=['rule_str'],
                color_continuous_scale=VIRIDIS_LIKE_SCALE,
                labels={'confidence': 'Confidence', 'lift': 'Lift'}
            )
            fig2.update_layout(height=400)
            fig2.update_traces(marker=dict(line=dict(width=1, color="white")))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown('<div class="info-box">No rules available to plot yet.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: FREQUENT ITEMSETS
# ------------------------------------------------------------------------------
with tab_itemsets:
    st.markdown(f'<div class="section-header">🔥 Discovered Frequent Itemsets (Top {top_n})</div>', unsafe_allow_html=True)

    filter_size = st.radio("Filter Itemset Length", ["All", "Single Products (1)", "Pairs (2)", "Triplets+ (3+)"], horizontal=True)

    filtered_itemsets = freq_itemsets.copy()
    if filter_size == "Single Products (1)":
        filtered_itemsets = filtered_itemsets[filtered_itemsets['item_count'] == 1]
    elif filter_size == "Pairs (2)":
        filtered_itemsets = filtered_itemsets[filtered_itemsets['item_count'] == 2]
    elif filter_size == "Triplets+ (3+)":
        filtered_itemsets = filtered_itemsets[filtered_itemsets['item_count'] >= 3]

    display_df = filtered_itemsets[['items', 'item_count', 'freq', 'support_pct']].head(top_n).rename(
        columns={
            'items': 'Product Combo / Itemset',
            'item_count': 'Items Count',
            'freq': 'Transaction Count',
            'support_pct': 'Support (%)'
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Support (%)": st.column_config.ProgressColumn(
                "Support (%)",
                help="Share of total transactions containing this itemset",
                format="%.2f%%",
                min_value=0,
                max_value=float(display_df['Support (%)'].max()) if not display_df.empty else 1,
            ),
        },
    )

# ------------------------------------------------------------------------------
# TAB 3: ASSOCIATION RULES & INTERACTIVE EXPLORER
# ------------------------------------------------------------------------------
with tab_rules:
    st.markdown(f'<div class="section-header">🔗 Association Rules Matrix (Sorted by {metric_sort.capitalize()})</div>', unsafe_allow_html=True)

    if not rules.empty:
        display_rules = rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].head(top_n).copy()
        display_rules['support'] = (display_rules['support'] * 100).round(2)
        display_rules['confidence'] = (display_rules['confidence'] * 100).round(2)
        display_rules['lift'] = display_rules['lift'].round(2)
        display_rules = display_rules.rename(columns={
            'antecedent_str': 'If Customer Buys (Antecedent)',
            'consequent_str': 'Recommend (Consequent)',
            'support': 'Support (%)',
            'confidence': 'Confidence (%)',
            'lift': 'Lift'
        })

        st.dataframe(
            display_rules,
            use_container_width=True,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence (%)", format="%.1f%%", min_value=0, max_value=100,
                ),
                "Lift": st.column_config.NumberColumn(
                    "Lift", format="%.2f ⚡", help="Values above 1 indicate positive correlation",
                ),
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:1.1rem;">🕸️ Interactive Rule Strength Matrix</div>', unsafe_allow_html=True)

        top_rules_vis = rules.head(15).copy()
        fig_parallel = px.parallel_categories(
            top_rules_vis,
            dimensions=['antecedent_str', 'consequent_str'],
            color="lift",
            color_continuous_scale=VIRIDIS_LIKE_SCALE,
            labels={'antecedent_str': 'Antecedent (If bought)', 'consequent_str': 'Consequent (Then bought)'}
        )
        fig_parallel.update_layout(height=450)
        st.plotly_chart(fig_parallel, use_container_width=True)

    else:
        st.markdown('<div class="info-box">No association rules found under current slider thresholds.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: REAL-TIME CROSS-SELLING ENGINE
# ------------------------------------------------------------------------------
with tab_recommender:
    st.markdown('<div class="section-header">🔮 Real-Time Customer Basket Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Simulate a live customer cart to generate optimized cross-selling recommendations</div>', unsafe_allow_html=True)

    cart_col1, cart_col2 = st.columns([1, 2])

    with cart_col1:
        default_items = ["pip fruit"] if "pip fruit" in all_products else [all_products[0]]
        user_cart = st.multiselect(
            "🛒 Select items in customer cart:",
            options=all_products,
            default=default_items
        )

    with cart_col2:
        if user_cart:
            st.markdown('<div class="section-header" style="font-size:1.05rem;">🎯 Smart Recommendations</div>', unsafe_allow_html=True)

            if not rules.empty:
                matched_rules = rules[
                    rules['antecedents'].apply(lambda x: set(x).issubset(set(user_cart)))
                ].sort_values(by='lift', ascending=False)

                rec_list = []
                for _, row in matched_rules.iterrows():
                    consequents = list(row['consequents'])
                    for c in consequents:
                        if c not in user_cart and c not in [r['product'] for r in rec_list]:
                            rec_list.append({
                                'product': c,
                                'confidence': row['confidence'],
                                'lift': row['lift'],
                                'reason': row['antecedent_str']
                            })

                if rec_list:
                    rec_df = pd.DataFrame(rec_list).head(5)

                    for idx, item in rec_df.iterrows():
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-title">🎁 {item['product']}</div>
                            <span class="badge badge-conf">✅ {item['confidence']*100:.1f}% confidence</span>
                            <span class="badge badge-lift">⚡ {item['lift']:.2f}x lift</span>
                            <span class="badge badge-reason">🛒 because of: {item['reason']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">No strong recommendation rules found for this specific cart combination under current slider settings.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">Please adjust sidebar slider thresholds to generate association rules.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Select one or more products above to view live cross-selling suggestions.</div>', unsafe_allow_html=True)