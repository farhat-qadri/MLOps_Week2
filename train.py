import csv
import json
import random
import math

# Set random seed
random.seed(42)

def load_data():
    """Load iris dataset from CSV"""
    data = []
    with open('iris.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row['sepal_length']), float(row['sepal_width']), 
                           float(row['petal_length']), float(row['petal_width'])]
                species = row['species']
                data.append((features, species))
            except ValueError:
                continue  # Skip rows with invalid data
    return data

def split_data(data, test_ratio=0.2):
    """Split data into train and test sets"""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

def euclidean_distance(point1, point2):
    """Calculate euclidean distance between two points"""
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def knn_predict(train_data, test_point, k=3):
    """Predict using k-nearest neighbors"""
    # Calculate distances to all training points
    distances = []
    for train_features, train_label in train_data:
        dist = euclidean_distance(test_point, train_features)
        distances.append((dist, train_label))
    
    # Sort by distance and get k nearest
    distances.sort()
    nearest = distances[:k]
    
    # Count votes
    votes = {}
    for _, label in nearest:
        votes[label] = votes.get(label, 0) + 1
    
    # Return most common label
    return max(votes, key=votes.get)

def calculate_accuracy(predictions, actual):
    """Calculate accuracy"""
    correct = sum(1 for p, a in zip(predictions, actual) if p == a)
    return correct / len(predictions)

def main():
    print("Loading iris dataset...")
    data = load_data()
    print(f"Loaded {len(data)} samples")
    
    # Split data
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    actual = []
    
    for test_features, test_label in test_data:
        pred = knn_predict(train_data, test_features)
        predictions.append(pred)
        actual.append(test_label)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, actual)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Count predictions by class
    pred_counts = {}
    for pred in predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    actual_counts = {}
    for label in actual:
        actual_counts[label] = actual_counts.get(label, 0) + 1
    
    # Save metrics to CSV
    with open('metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy'])
        writer.writerow(['KNN', accuracy])
    
    # Save predictions to CSV
    with open('predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Actual', 'Predicted', 'Correct'])
        for a, p in zip(actual, predictions):
            writer.writerow([a, p, 'Yes' if a == p else 'No'])
    
    # Save model info (just the training data as JSON)
    model_data = {
        'algorithm': 'KNN',
        'k': 3,
        'training_samples': len(train_data),
        'accuracy': accuracy,
        'predictions_count': pred_counts,
        'actual_count': actual_counts
    }
    
    with open('model_weights.h5', 'w') as f:  # Using .h5 name as requested
        json.dump(model_data, f, indent=2)
    
    print("\nFiles created:")
    print("- metrics.csv")
    print("- predictions.csv") 
    print("- model_weights.h5 (model info)")
    print(f"\nFinal Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
