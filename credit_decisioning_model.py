import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report

# 1. Load data (placeholder: replace with your actual file path)
df = pd.read_csv('data/loan_data.csv')

# 2. Business-driven filtering: keep accounts aged 6–24 months
df = df[(df['account_age_months'] >= 6) & (df['account_age_months'] <= 24)]

# 3. Drop non-model columns
df = df.drop(columns=['loan_id', 'origination_date'], errors='ignore')

# 4. Encode categorical variables
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 5. Prepare features and target
X = df.drop(columns=['charge_off_flag'])
y = df['charge_off_flag']

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Feature importance
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("Top 10 features by coefficient magnitude:")
print(coefs.head(10))
