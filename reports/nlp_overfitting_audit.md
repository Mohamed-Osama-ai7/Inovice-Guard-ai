# NLP Overfitting Audit

## Summary

- Best candidate: Character TF-IDF + Logistic Regression
- Train F1: 1.0000
- Validation F1: 0.9072
- Final unseen F1: 0.9347
- OOD F1: 0.8548
- Adversarial F1: 0.8646
- High-risk recall on final test: 0.7600
- Overfitting level: MODERATE

## Leakage checks

- Exact duplicate count: 0
- Near duplicate count: 0
- Train/test exact overlap: 0
- Train/final unseen exact overlap: 0
- Train/OOD exact overlap: 0
- Train/adversarial exact overlap: 0
- Train/test group overlap: 0

## Interpretation

- All held-out evaluation sets are frozen and never used for model fitting, hyperparameter tuning, or threshold selection.
- The train/validation/test split is semantic-group aware, so examples from the same underlying intent are kept together.
- Final model selection therefore prioritizes OOD/adversarial robustness and final unseen generalization rather than training accuracy.