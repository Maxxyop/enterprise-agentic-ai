def calculate_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if labels else 0.0

def calculate_precision(predictions, labels):
    true_positive = sum(p == l == 1 for p, l in zip(predictions, labels))
    predicted_positive = sum(p == 1 for p in predictions)
    return true_positive / predicted_positive if predicted_positive else 0.0

def calculate_recall(predictions, labels):
    true_positive = sum(p == l == 1 for p, l in zip(predictions, labels))
    actual_positive = sum(l == 1 for l in labels)
    return true_positive / actual_positive if actual_positive else 0.0

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

def evaluate_metrics(predictions, labels):
    accuracy = calculate_accuracy(predictions, labels)
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    f1_score = calculate_f1_score(precision, recall)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }