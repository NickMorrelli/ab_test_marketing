# 📧 A/B Test: Email Marketing Campaign Analysis

A complete end-to-end A/B test analysis pipeline using the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail). This project simulates a promotional email campaign and measures its impact on **conversion rate** and **revenue per customer** using both **frequentist** and **Bayesian** statistical methods.

---

## 🎯 Business Question

> *Did sending a promotional email campaign to existing customers increase conversion rates and revenue compared to customers who did not receive the email?*

---

## 📁 Project Structure

```
ab_test_marketing/
├── data/                        # Place OnlineRetail.xlsx here
├── outputs/                     # Generated charts saved here
├── src/
│   ├── data_prep.py             # Load, clean, engineer A/B groups
│   ├── frequentist.py           # Chi-square, Mann-Whitney U, Welch's t-test
│   ├── bayesian.py              # Beta-Binomial & Log-Normal Monte Carlo
│   └── visualizations.py       # All charts and summary dashboard
├── main.py                      # Run the full pipeline
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**UCI Online Retail Dataset** — real-world UK e-commerce transactions (Dec 2010 – Dec 2011).

- ~500,000 transactions  
- ~4,000 unique customers  
- Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

**Download:** https://archive.ics.uci.edu/dataset/352/online+retail  
Place `OnlineRetail.xlsx` in the `/data/` folder before running.

---

## 🧪 Experimental Design

| | Detail |
|---|---|
| **Unit of randomization** | Customer |
| **Eligible population** | Customers with ≥1 purchase before June 1, 2011 |
| **Control** | Did not receive promotional email |
| **Treatment** | Received promotional email |
| **Split** | 50 / 50 random assignment |
| **Campaign lift (simulated)** | +8pp conversion rate, +15% AOV |
| **Primary metrics** | Conversion rate, Revenue per customer |
| **Secondary metric** | Average order value (converters only) |

---

## 📐 Statistical Methods

### Frequentist
| Test | Metric | Why |
|---|---|---|
| Chi-Square | Conversion rate | Test independence of group and conversion |
| Mann-Whitney U | Revenue per customer | Non-parametric; handles right-skewed revenue |
| Welch's t-test | Avg order value | Compare means for converters only |

### Bayesian
| Model | Metric | Why |
|---|---|---|
| Beta-Binomial conjugate | Conversion rate | Closed-form posterior; interpretable probability |
| Log-Normal Monte Carlo | Revenue per customer | Handles skewed, zero-heavy distributions |

Bayesian outputs include:
- Posterior mean + 95% credible interval
- P(Treatment > Control)
- Expected lift and expected loss

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/ab_test_marketing.git
cd ab_test_marketing
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**  
Get `OnlineRetail.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it in `/data/`.

**4. Run the pipeline**
```bash
python main.py
```

Charts will be saved to `/outputs/`.

---

## 📈 Output Charts

| File | Description |
|---|---|
| `01_conversion_rate.png` | Bar chart with 95% Wilson CI error bars |
| `02_revenue_distribution.png` | KDE of post-campaign revenue |
| `03_bayesian_conversion_posterior.png` | Beta posterior distributions |
| `04_bayesian_revenue_posterior.png` | Revenue posterior distributions |
| `05_summary_dashboard.png` | 2×2 summary dashboard |

---

## 🛠 Tech Stack

- **Python 3.14+**
- `pandas` — data wrangling
- `numpy` — numerical computing
- `scipy` — statistical tests
- `matplotlib` — visualizations

---

## 💡 Key Concepts Demonstrated

- A/B test design and experimental setup
- Feature engineering from raw transaction data
- Frequentist hypothesis testing (p-values, effect sizes)
- Bayesian inference (conjugate priors, credible intervals, expected loss)
- Marketing funnel analysis (conversion → AOV → revenue)
- Data visualization for executive-ready reporting

---

## 👤 Author

Built as part of a data science portfolio project.  
Background: 15+ years in Marketing Analytics | SQL | Python | Statistical Modeling
