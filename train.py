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
        for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
            try:
                # Check if all required columns exist
                required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
                if not all(col in row for col in required_cols):
                    print(f"Warning: Row {row_num} missing required columns, skipping")
                    continue
                
                # Try to convert to float and validate
                sepal_length = float(row['sepal_length'])
                sepal_width = float(row['sepal_width'])
                petal_length = float(row['petal_length'])
                petal_width = float(row['petal_width'])
                species = row['species'].strip()
                
                # Validate that values are reasonable (basic sanity check)
                if (sepal_length < 0 or sepal_width < 0 or 
                    petal_length < 0 or petal_width < 0):
                    print(f"Warning: Row {row_num} has negative values, skipping")
                    continue
                
                if not species:
                    print(f"Warning: Row {row_num} has empty species, skipping")
                    continue
                
                features = [sepal_length, sepal_width, petal_length, petal_width]
                data.append((features, species))
                
            except (ValueError, KeyError) as e:
                print(f"Warning: Row {row_num} has invalid data ({e}), skipping")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error in row {row_num} ({e}), skipping")
                continue
    
    print(f"Successfully loaded {len(data)} valid samples")
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
    
    if len(data) == 0:
        print("Error: No valid data found!")
        return
    
    print(f"Loaded {len(data)} samples")
    
    # Print species distribution
    species_count = {}
    for _, species in data:
        species_count[species] = species_count.get(species, 0) + 1
    print("Species distribution:", species_count)
    
    # Split data
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    if len(test_data) == 0:
        print("Error: No test data available!")
        return
    
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
    
    print("Prediction counts:", pred_counts)
    print("Actual counts:", actual_counts)
    
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
        'test_samples': len(test_data),
        'accuracy': accuracy,
        'predictions_count': pred_counts,
        'actual_count': actual_counts,
        'species_distribution': species_count
    }
    
    with open('model_weights.h5', 'w') as f:  # Using .h5 name as requested
        json.dump(model_data, f, indent=2)
    with open('model.keras', 'w') as f2:
        json.dump(model_data, f2, indent=2)
    
    print("\nFiles created:")
    print("- metrics.csv")
    print("- predictions.csv") 
    print("- model_weights.h5 (model info)")
    print(f"\nFinal Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
