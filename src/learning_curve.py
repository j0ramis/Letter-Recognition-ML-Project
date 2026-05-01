import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import os

def main():
    # 1. Load the data
    try:
        df = pd.read_csv('../data/letter-recognition.csv')
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print("Error: Ensure letter-recognition.csv is in the '../data/' folder.")
        return

    # 2. Split into Train, Validation, and Test sets
    X = df.drop('letter', axis=1)
    y = df['letter']
    
    # First split: 80% Train/Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Second split: separate the Temp data into Training and Validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

    # 3. Initialize the MLP Classifier for manual epoch training
    # Notice we don't set max_iter because we will control the loop manually
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, solver='adam')
    
    classes = np.unique(y)
    epochs = 100 # Adjust this if you want to see a longer or shorter timeline
    
    train_loss = []
    val_loss = []

    print(f"Training MLP across {epochs} epochs to track loss...")

    # 4. Train the model one epoch at a time
    for epoch in range(epochs):
        # partial_fit trains the model for just one iteration
        mlp.partial_fit(X_train, y_train, classes=classes)
        
        # Predict probabilities to calculate the loss
        train_probs = mlp.predict_proba(X_train)
        val_probs = mlp.predict_proba(X_val)
        
        # Calculate and store the log loss
        train_loss.append(log_loss(y_train, train_probs))
        val_loss.append(log_loss(y_val, val_probs))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed.")

    # 5. Create the Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot both lines on the same visual
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', color='orange', linewidth=2)
    
    # Add titles, labels, and visual formatting
    plt.title('Learning Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the image
    plt.tight_layout()
    plt.savefig('../images/learning_curve.png')
    print("\nSuccess! 'learning_curve.png' has been saved to your images folder.")

if __name__ == "__main__":
    main()
