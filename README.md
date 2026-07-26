# 🛒 Retail Insights: Market Basket Analytics Engine

An enterprise-grade, interactive **Market Basket Analysis Dashboard** powered by **PySpark**, **MLxtend**, and **Streamlit**. 

This application utilizes the **FP-Growth (Frequent Pattern Growth)** algorithm to discover hidden customer purchasing patterns, frequent product combinations, and strong association rules from retail transactional data.

🚀 **Live Interactive App:** [Market Basket Analytics Dashboard](https://marketbasketdashboard.streamlit.app/)

---

## 📌 Features

- **⚡ Fast FP-Growth Engine:** High-performance pattern mining capable of handling thousands of basket transactions.
- **🎛️ Real-Time Dynamic Parameter Control:** Interactively tune `minSupport` and `minConfidence` thresholds via a smooth sidebar slider.
- **💡 Automated Business Insights:** Generates plain-English natural language takeaways explaining high-lift association rules for executive decision-making.
- **📊 Interactive Visualizations:** Built-in **Plotly** horizontal bar charts, scatter plots (Lift vs. Confidence), and parallel coordinates diagrams.
- **🔮 Real-Time Cross-Selling Engine:** Simulate a live customer cart to generate instant product recommendations with confidence and lift metrics.
- **📱 Fully Responsive UI:** Styled with custom CSS cards, metric badges, and organized multi-tab navigation.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Frontend / Dashboard Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** `pandas`, `numpy`
- **Frequent Pattern Mining:** `mlxtend` / `pyspark`
- **Interactive Data Visualization:** `plotly`

---

## 📂 Dataset Overview

The dashboard is configured for transactional grocery datasets (wide or long format).

- **Total Transactions:** ~9,835 customer baskets
- **Unique Products:** 169 retail items
- **Top Frequent Items:** Whole milk, Other vegetables, Rolls/buns, Soda, Yogurt

---

## 📐 Mathematical Metrics Explained

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **Support** | $\text{Support}(A \Rightarrow B) = \frac{\text{Freq}(A, B)}{N}$ | Fraction of total transactions containing both item A and item B. |
| **Confidence** | $\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}$ | Likelihood that item B is purchased when item A is purchased. |
| **Lift** | $\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$ | Strength of the rule over random chance. $\text{Lift} > 1$ indicates strong positive affinity. |

---

## 🚀 Local Installation & Setup

Follow these steps to run the dashboard locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
