import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("breast_cancer_prediction.csv")

df = df[df['Gender'] == 'Female'].copy()

df.drop(columns=['Annual_Income_USD', 'Patient_ID', 'Mammogram_Result', 'Biopsy_Result',
                  'Cancer_Stage', 'Tumor_Size_cm', 'Lymph_Node_Involvement', 'Gender'], inplace=True)

df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna(df['Alcohol_Consumption'].mode()[0])
df['Physical_Activity'] = df['Physical_Activity'].fillna(df['Physical_Activity'].mode()[0])
df['Hormone_Therapy'] = df['Hormone_Therapy'].fillna(df['Hormone_Therapy'].mode()[0])

binary_cols = ['Family_History', 'Smoking', 'Alcohol_Consumption', 'Hormone_Therapy', 'Diabetes']
for col in binary_cols:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

df['Genetic_Mutation'] = df['Genetic_Mutation'].map({'Negative': 0, 'Positive': 1})
df = pd.get_dummies(df, columns=['Physical_Activity'], drop_first=True)
df['Menopause_Status'] = df['Menopause_Status'].map({'Post': 0, 'Pre': 1})
df['Physical_Activity_Low'] = df['Physical_Activity_Low'].map({False: 0, True: 1})
df['Physical_Activity_Moderate'] = df['Physical_Activity_Moderate'].map({False: 0, True: 1})

df = pd.get_dummies(df, columns=['Breastfeeding_History'], drop_first=True, dtype=int)

X = df.drop(columns=['Cancer'])
y = df['Cancer']

# save the exact column order the model expects - the Flask app must build rows in this order
feature_columns = X.columns.tolist()
print("Feature columns (order matters):", feature_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(feature_columns, 'feature_columns.joblib')

print("\nSaved: model.joblib, scaler.joblib, feature_columns.joblib")
