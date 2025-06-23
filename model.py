import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_and_save_model():
    # Modèle fictif simple
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, 'model.joblib')
    print("Modèle entraîné et sauvegardé.")

def predict(input_data):
    model = joblib.load('model.joblib')
    prediction = model.predict([input_data])
    print(f"Prédiction : {prediction[0]}")
    return prediction[0]

if __name__ == "__main__":
    train_and_save_model()
    predict([3, 4])