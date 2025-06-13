import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_csv("Sleep_Health.csv")

# Pilih fitur dan target
X = df[[
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps'
]]
y = df['Sleep Disorder']

# Encode kategorikal
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Simpan kategori unik untuk UI
unique_categories = {
    'gender': sorted(df['Gender'].dropna().unique().tolist()),
    'occupation': sorted(df['Occupation'].dropna().unique().tolist()),
    'bmi': sorted(df['BMI Category'].dropna().unique().tolist())
}

# Ubah kategorikal di X jadi numerik
X['Gender'] = X['Gender'].astype(str)
X['Occupation'] = X['Occupation'].astype(str)
X['BMI Category'] = X['BMI Category'].astype(str)

X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Latih model ANN
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Simpan model dan artefak lain
joblib.dump(model, 'ann_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(unique_categories, 'unique_categories.pkl')
joblib.dump(conf_matrix, 'conf_matrix.pkl')
joblib.dump(report_dict, 'classification_report.pkl')

print("✅ Model dan encoder berhasil disimpan.")
