# Data Leakage Audit Report

This document confirms the rigorous auditing of feature engineering pipelines across the InvoiceGuard AI Enterprise Platform to ensure zero data leakage.

## Core Principle
For every prediction at time $T$, all features are constructed strictly using information available at $T$ or earlier. Targets representing future behavior ($>T$) are strictly separated and only used during training.

## 1. Retail Intelligence Pipeline (`src/retail_pipeline.py`)

### Temporal Snapshot Strategy
The retail dataset utilizes a rolling snapshot window strategy (`build_temporal_dataset`).
- **Feature Window**: $[ \text{first\_purchase\_date}, T )$
- **Target Window**: $[ T, T + \text{forward\_days} )$

### Audit by Feature
| Feature | Allowed Timestamp | Source | Leakage Risk Assessment | Result |
|---------|-------------------|--------|-------------------------|--------|
| `recency_days` | $< T$ | `snapshot_date - last_purchase_date` | Safe. `last_purchase_date` is filtered to $< T$. | PASS |
| `frequency` | $< T$ | Count of invoices | Safe. Filtered to $< T$. | PASS |
| `monetary` | $< T$ | Sum of revenue | Safe. Filtered to $< T$. | PASS |
| `cancellation_rate` | $< T$ | Cancellations / Frequency | Safe. Derived from historical data only. | PASS |
| `revenue_90d` | $[T-90, T)$ | Sum of revenue in 90 days prior | Safe. Bounded window strictly before $T$. | PASS |

### Resampling and Imbalance
The retail pipeline does not use synthetic oversampling (e.g., SMOTE) before temporal splitting. Imbalance is handled at the algorithm level (via `class_weight='balanced'`) to prevent cross-contamination.

---

## 2. Invoice Payment Pipeline (`src/pipeline.py`)

### Chronological Strategy
Invoice payment features are constructed using a `shift(1)` logic grouped by customer, sorted chronologically by `invoice_date`.

### Audit by Feature
| Feature | Allowed Timestamp | Source | Leakage Risk Assessment | Result |
|---------|-------------------|--------|-------------------------|--------|
| `invoice_amount` | $T$ | Current invoice | Safe. Available at creation. | PASS |
| `days_to_due` | $T$ | `due_date - invoice_date` | Safe. Derived from contractual dates. | PASS |
| `prior_late_count` | $< T$ | `shift(1)` sum of prior late payments | Safe. Uses strictly previous invoice outcomes. | PASS |
| `prior_avg_delay` | $< T$ | `shift(1)` mean of prior delay days | Safe. Uses strictly previous invoice outcomes. | PASS |

### Target Variable
The target (`late_payment` and `delay_days`) relies on the `payment_date`. The pipeline explicitly drops `payment_date` and any derived targets from the feature set before training (`add_features` L124: `leakage={'late_payment','delay_days','delay_days_positive',m.payment_date}`).

## Conclusion
**STATUS: PASS**
Both pipelines have been audited and no instances of temporal or structural data leakage were found.
