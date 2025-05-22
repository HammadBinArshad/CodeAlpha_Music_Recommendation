
# MUSIC RECOMMENDATION SYSTEM - CodeAlpha Internship

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('D:\CODE ALPHA INTERNSHIP\SOURCECODES-INTERNSHIPS\data-MUSIC.csv')

# Create binary label: 'replayed' = 1 if popularity >= 50
df['replayed'] = df['popularity'].apply(lambda x: 1 if x >= 50 else 0)

# Drop unnecessary columns
columns_to_drop = ['id', 'name', 'artists', 'release_date', 'popularity']
df_cleaned = df.drop(columns=columns_to_drop)

# Split into features and target
X = df_cleaned.drop(columns=['replayed'])
y = df_cleaned['replayed']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with class weights
log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

# Random Forest on 20,000 sample rows to reduce training time
sample_df = df_cleaned.sample(n=20000, random_state=42)
X_sample = sample_df.drop(columns=['replayed'])
y_sample = sample_df['replayed']
X_sample_train, X_sample_test, y_sample_train, y_sample_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42)
rf_model.fit(X_sample_train, y_sample_train)
y_pred_rf = rf_model.predict(X_sample_test)

# Evaluate Random Forest
print("Random Forest Report:")
print(classification_report(y_sample_test, y_pred_rf))

# Plot confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_sample_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Oranges', xticklabels=['Not Replayed', 'Replayed'], yticklabels=['Not Replayed', 'Replayed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()
