# Credit Decisioning Model

This project demonstrates a logistic regression approach to predict loan charge-off probability using engineered features from borrower behavioral and transactional data.

## Features (Data Censored)

- Data cleaning and filtering based on business logic (e.g., selecting loans with 6–20+ month age)
- Logistic regression model for binary classification
- Handling of class imbalance awareness (~10%+ default rate)
- Evaluation using ROC-AUC, precision, and recall
- Interpretation of model results through coefficients

## Notes

- Data is not included due to confidentiality.
- This notebook is designed for reproducibility demonstration.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
