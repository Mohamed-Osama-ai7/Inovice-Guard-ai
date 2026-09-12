from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo" / "demo_invoices.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 12000
customers = rng.integers(1, 1801, size=n)
invoice_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1460, n), unit="D")
terms = rng.choice([15, 30, 45, 60, 90], n, p=[.10,.40,.22,.18,.10])
due_date = invoice_date + pd.to_timedelta(terms, unit="D")
amount = np.round(np.exp(rng.normal(8.0, 1.0, n)), 2)
industry = rng.choice(["Construction","Retail","IT","Manufacturing","Services"], n)
company_size = rng.choice(["Small","Medium","Large"], n, p=[.42,.40,.18])
payment_method = rng.choice(["Bank Transfer","Cheque","Card","Cash"], n, p=[.50,.22,.20,.08])
segment = rng.choice(["SME","Mid-Market","Enterprise"], n, p=[.55,.32,.13])

# Customer-specific latent reliability, then build observable historical features.
reliability = rng.normal(0, 1, 1801)
base = reliability[customers] + 0.0025 * (amount - amount.mean()) + 0.012 * (terms - 30)
industry_effect = pd.Series(industry).map({"Construction":.45,"Retail":.05,"IT":-.25,"Manufacturing":.18,"Services":.00}).to_numpy()
size_effect = pd.Series(company_size).map({"Small":.20,"Medium":0.0,"Large":-.20}).to_numpy()
method_effect = pd.Series(payment_method).map({"Bank Transfer":-.12,"Cheque":.42,"Card":-.25,"Cash":-.05}).to_numpy()
logit = base + industry_effect + size_effect + method_effect
p_late = 1 / (1 + np.exp(-logit))
late = (rng.random(n) < p_late).astype(int)
# Delay days are zero when on time, right-skewed when late.
delay = np.where(late, np.clip(rng.gamma(shape=2.2, scale=5.0, size=n) + 3 + 3*np.maximum(logit,0), 1, 120), -rng.integers(0,5,n))
payment_date = due_date + pd.to_timedelta(delay, unit="D")

# Historical features available before current invoice are generated from latent behavior.
prior_count = rng.integers(0, 16, n)
prior_late_ratio = np.clip(0.18 + 0.20*reliability[customers] + rng.normal(0, .07, n), 0, 1)
prior_late_count = np.round(prior_count * prior_late_ratio).astype(int)
prior_avg_delay = np.where(prior_count>0, np.clip(2 + 13*prior_late_ratio + rng.normal(0, 2, n), 0, 60), 0)
outstanding = np.round(amount * np.clip(rng.uniform(.0, 2.2, n), 0, None), 2)

# Small noise and some missingness.
df = pd.DataFrame({
    "invoice_id": [f"INV-{i:06d}" for i in range(n)],
    "customer_id": [f"C-{c:04d}" for c in customers],
    "invoice_date": invoice_date,
    "due_date": due_date,
    "payment_date": payment_date,
    "invoice_amount": amount,
    "industry": industry,
    "company_size": company_size,
    "payment_method": payment_method,
    "customer_segment": segment,
    "prior_invoice_count": prior_count,
    "prior_late_count": prior_late_count,
    "prior_late_ratio": prior_late_ratio,
    "prior_avg_delay": prior_avg_delay,
    "outstanding_amount": outstanding,
})

for col, frac in [("industry",.015),("company_size",.01),("payment_method",.012),("invoice_amount",.01)]:
    idx = rng.choice(n, size=int(n*frac), replace=False)
    df.loc[idx, col] = np.nan

OUT.write_text(df.to_csv(index=False), encoding="utf-8")
print(f"Wrote {len(df):,} rows to {OUT}")
