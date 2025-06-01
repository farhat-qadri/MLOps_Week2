import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """Load and preprocess the iris dataset"""
    print("Loading iris dataset...")
    
    # Load the data
    df = pd.read_csv('iris.csv')
    
    # Remove any potential duplicates
    df = df.drop_duplicates()
    
    # Basic data info
    print(f"Dataset shape: {df.shape}")
    print(f"Species distribution:\n{df['species'].value_counts()}")
    
    # Separate features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = df['species'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

def train_traditional_models(X_train, X_test, y_train, y_test):
    """Train traditional ML models"""
    print("\n=== Training Traditional ML Models ===")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, f'{name.lower()}_model.pkl')
    
    return results

def train_neural_network(X_train, X_test, y_train, y_test):
    """Train a simple neural network"""
    print("\n=== Training Neural Network ===")
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"Neural Network Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('model_weights.h5')
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    return {
        'accuracy': test_accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'history': history_dict
    }

def save_metrics(traditional_results, nn_results, y_test, label_encoder):
    """Save detailed metrics to CSV"""
    print("\n=== Saving Metrics ===")
    
    # Combine all results
    all_results = {**traditional_results, 'NeuralNetwork': nn_results}
    
    # Create metrics summary
    metrics_summary = []
    
    for model_name, results in all_results.items():
        y_pred = results['predictions']
        accuracy = results['accuracy']
        
        # Calculate per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Add overall metrics
        metrics_summary.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision_Macro': report['macro avg']['precision'],
            'Recall_Macro': report['macro avg']['recall'],
            'F1_Macro': report['macro avg']['f1-score'],
            'Precision_Weighted': report['weighted avg']['precision'],
            'Recall_Weighted': report['weighted avg']['recall'],
            'F1_Weighted': report['weighted avg']['f1-score']
        })
        
        # Save detailed classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{model_name.lower()}_classification_report.csv')
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                           index=[f'True_{cls}' for cls in label_encoder.classes_],
                           columns=[f'Pred_{cls}' for cls in label_encoder.classes_])
        cm_df.to_csv(f'{model_name.lower()}_confusion_matrix.csv')
    
    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv('metrics.csv', index=False)
    
    print("Metrics saved to:")
    print("- metrics.csv (summary)")
    print("- *_classification_report.csv (detailed reports)")
    print("- *_confusion_matrix.csv (confusion matrices)")

def save_predictions(traditional_results, nn_results, X_test, y_test, label_encoder):
    """Save predictions and probabilities"""
    print("\n=== Saving Predictions ===")
    
    # Create predictions dataframe
    predictions_data = {
        'true_label_encoded': y_test,
        'true_label': label_encoder.inverse_transform(y_test)
    }
    
    # Add feature values
    for i, feature in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
        predictions_data[feature] = X_test[:, i]
    
    # Add predictions from all models
    all_results = {**traditional_results, 'NeuralNetwork': nn_results}
    
    for model_name, results in all_results.items():
        predictions_data[f'{model_name}_prediction_encoded'] = results['predictions']
        predictions_data[f'{model_name}_prediction'] = label_encoder.inverse_transform(results['predictions'])
        
        # Add probabilities for each class
        proba = results['probabilities']
        for i, class_name in enumerate(label_encoder.classes_):
            predictions_data[f'{model_name}_prob_{class_name}'] = proba[:, i]
    
    # Save to CSV
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv('predictions.csv', index=False)
    
    print("Predictions saved to predictions.csv")

def main():
    print("=== Iris Classification Training Pipeline ===")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()
    
    # Train traditional models
    traditional_results = train_traditional_models(X_train, X_test, y_train, y_test)
    
    # Train neural network
    nn_results = train_neural_network(X_train, X_test, y_train, y_test)
    
    # Save metrics
    save_metrics(traditional_results, nn_results, y_test, label_encoder)
    
    # Save predictions
    save_predictions(traditional_results, nn_results, X_test, y_test, label_encoder)
    
    print("\n=== Training Complete! ===")
    print("\nOutput files created:")
    print("1. metrics.csv - Model performance summary")
    print("2. model_weights.h5 - Neural network model")
    print("3. predictions.csv - All model predictions")
    print("4. training_history.json - Neural network training history")
    print("5. *_model.pkl - Traditional ML models")
    print("6. *_classification_report.csv - Detailed classification reports")
    print("7. *_confusion_matrix.csv - Confusion matrices")
    print("8. scaler.pkl - Feature scaler")
    print("9. label_encoder.pkl - Label encoder")
    
    # Print best model
    all_results = {**traditional_results, 'NeuralNetwork': nn_results}
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

if __name__ == "__main__":
    main()