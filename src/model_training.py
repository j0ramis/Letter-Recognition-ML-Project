import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

def main():
    # 1. Load Data
    df = pd.read_csv('../data/letter-recognition.csv')
    df.columns = df.columns.str.strip() # Preprocessing: strip trailing spaces
    
    # 2. Split Data
    X = df.drop('letter', axis=1)
    y = df['letter']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Support Vector Machine
    print("Training SVM...")
    start_time = time.time()
    svm_model = SVC(kernel='rbf', C=10, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_predictions)
    print(f"SVM Accuracy: {svm_acc:.4f} (Time: {time.time() - start_time:.2f} seconds)")

    # 4. Train Multi-Layer Perceptron
    print("\nTraining MLP...")
    start_time = time.time()
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_predictions)
    print(f"MLP Accuracy: {mlp_acc:.4f} (Time: {time.time() - start_time:.2f} seconds)")

if __name__ == "__main__":
    main()
