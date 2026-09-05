import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 14)  
pd.set_option('display.width', 100) 
df = pd.read_csv("breast_cancer_prediction.csv")
#print(df)
#print("BEFORE")
#print(df.info())
#print(df.describe())
#print(df.head())
#print(df.tail())
df = df[df['Gender'] == 'Female'].copy()
#print(df.head())
df.drop(columns=['Annual_Income_USD','Patient_ID','Mammogram_Result','Biopsy_Result','Cancer_Stage','Tumor_Size_cm','Lymph_Node_Involvement','Gender'], inplace=True)
#print("AFTER")
#print(df.info())
#print(df.head())
df ['BMI'] =df['BMI'].fillna(df['BMI'].median())
df ['Alcohol_Consumption'] =df['Alcohol_Consumption'].fillna(df['Alcohol_Consumption'].mode()[0])
df['Physical_Activity'] = df['Physical_Activity'].fillna(df['Physical_Activity'].mode()[0])
df['Hormone_Therapy'] =df['Hormone_Therapy'].fillna(df['Hormone_Therapy'].mode()[0])
#df['Breastfeeding_History'] =df['Breastfeeding_History'].fillna(df['Breastfeeding_History'].mode()[0])
#print(df.head())
#print(df.info())
binary_cols = ['Family_History', 'Smoking', 'Alcohol_Consumption', 
               'Hormone_Therapy', 'Diabetes']

for col in binary_cols:
    df[col] = df[col].map({'No': 0, 'Yes': 1})
#df['Physical_Activity'] = df['Physical_Activity'].map({'Low': 0, 'Moderate': 1, 'High': 2})
df['Genetic_Mutation'] = df['Genetic_Mutation'].map({'Negative': 0, 'Positive': 1})
df = pd.get_dummies(df, columns=['Physical_Activity'], drop_first=True)
df['Menopause_Status'] = df['Menopause_Status'].map({'Post': 0, 'Pre': 1})
df['Physical_Activity_Low'] = df['Physical_Activity_Low'].map({False: 0, True: 1})
df['Physical_Activity_Moderate'] = df['Physical_Activity_Moderate'].map({False: 0, True: 1})
#print(df.head())
#print(df.info())
#print("Columns list (exact names):")
#print(df.columns.tolist())

#print("\nUnique values in Breastfeeding_History BEFORE fillna:")
#print(df['Breastfeeding_History'].unique())

#print("\nMissing count BEFORE fillna:", df['Breastfeeding_History'].isnull().sum())

#mode_val = df['Breastfeeding_History'].mode()[0]
#print("\nMode value:", repr(mode_val))

#df['Breastfeeding_History'] = df['Breastfeeding_History'].fillna(mode_val)


binary_cols = ['Family_History', 'Smoking', 'Alcohol_Consumption', 
               'Hormone_Therapy', 'Diabetes']


df = pd.get_dummies(df, columns=['Breastfeeding_History'], drop_first=True, dtype=int)
#print(df.head())
#print(df.info())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['Cancer'])
y = df['Cancer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
from sklearn.metrics import classification_report

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(df['Cancer'].value_counts())
print(df['Cancer'].value_counts(normalize=True))  
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
import joblib

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train_resampled, y_train_resampled)

joblib.dump(final_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')

print("Saved: model.joblib, scaler.joblib, feature_columns.joblib")