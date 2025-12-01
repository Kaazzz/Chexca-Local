"""
Mock data generator for demonstration purposes
"""
import numpy as np
import random
from typing import Dict, List, Tuple
import config


def generate_mock_predictions() -> Dict[str, float]:
    """
    Generate realistic mock predictions for all 14 NIH diseases

    Returns:
        Dictionary mapping disease names to probabilities
    """
    # Create realistic probability distribution
    # Most diseases should have low probability, 1-3 should be elevated
    predictions = {}

    # Randomly select 1-3 diseases to have elevated probabilities
    num_positive = random.randint(1, 3)
    positive_diseases = random.sample(config.DISEASE_CLASSES, num_positive)

    for disease in config.DISEASE_CLASSES:
        if disease in positive_diseases:
            # Elevated probability for suspected diseases
            prob = random.uniform(0.45, 0.85)
        else:
            # Low probability for unlikely diseases
            prob = random.uniform(0.05, 0.35)

        predictions[disease] = prob

    return predictions


def generate_mock_co_occurrence(predictions: Dict[str, float],
                               primary_disease: str = None,
                               secondary_disease: str = None) -> np.ndarray:
    """
    Generate realistic co-occurrence matrix with emphasis on detected diseases

    Args:
        predictions: Disease predictions
        primary_disease: Primary detected disease (gets highest values)
        secondary_disease: Secondary detected disease (gets second highest values)

    Returns:
        14x14 co-occurrence matrix
    """
    n = len(config.DISEASE_CLASSES)
    co_occurrence = np.zeros((n, n))

    probs = np.array(list(predictions.values()))

    # Get indices of primary and secondary diseases
    primary_idx = None
    secondary_idx = None
    if primary_disease:
        try:
            primary_idx = config.DISEASE_CLASSES.index(primary_disease)
        except ValueError:
            pass
    if secondary_disease:
        try:
            secondary_idx = config.DISEASE_CLASSES.index(secondary_disease)
        except ValueError:
            pass

    # Common disease co-occurrences (medical knowledge)
    common_pairs = [
        ('Atelectasis', 'Pneumonia'),
        ('Effusion', 'Cardiomegaly'),
        ('Infiltration', 'Pneumonia'),
        ('Consolidation', 'Pneumonia'),
        ('Edema', 'Cardiomegaly'),
        ('Pneumothorax', 'Emphysema'),
    ]

    for i in range(n):
        for j in range(n):
            disease_i = config.DISEASE_CLASSES[i]
            disease_j = config.DISEASE_CLASSES[j]

            if i == j:
                # Diagonal = disease probability itself
                if i == primary_idx:
                    co_occurrence[i][j] = 0.98  # Highest for primary (Pneumonia)
                elif i == secondary_idx:
                    co_occurrence[i][j] = 0.72  # Second highest for secondary (Infiltration)
                else:
                    co_occurrence[i][j] = probs[i] * 0.25  # Lower for others
            else:
                # Off-diagonal elements
                is_common_pair = (
                    (disease_i, disease_j) in common_pairs or
                    (disease_j, disease_i) in common_pairs
                )

                # Special handling for primary-secondary pair
                if (i == primary_idx and j == secondary_idx) or (i == secondary_idx and j == primary_idx):
                    co_occurrence[i][j] = 0.92  # Second highest (Pneumonia-Infiltration)
                # High correlation if involves primary (Pneumonia) but not secondary
                elif i == primary_idx or j == primary_idx:
                    if is_common_pair:
                        co_occurrence[i][j] = random.uniform(0.55, 0.65)  # Lower than Infiltration diagonal
                    else:
                        co_occurrence[i][j] = random.uniform(0.35, 0.50)
                # Moderate correlation if involves secondary (Infiltration) but not primary
                elif i == secondary_idx or j == secondary_idx:
                    if is_common_pair:
                        co_occurrence[i][j] = random.uniform(0.40, 0.50)
                    else:
                        co_occurrence[i][j] = random.uniform(0.20, 0.35)
                # Normal correlation for other pairs
                elif is_common_pair:
                    base = (probs[i] + probs[j]) / 2
                    co_occurrence[i][j] = min(0.35, base * random.uniform(1.0, 1.2))
                else:
                    # Low correlation for unrelated diseases
                    co_occurrence[i][j] = random.uniform(0.05, 0.15)

    return co_occurrence


def get_mock_analysis_result(image_path: str = None) -> Dict:
    """
    Generate complete mock analysis result

    Args:
        image_path: Path to uploaded image (used for original image reference)

    Returns:
        Complete analysis result matching the real API response
    """
    # Generate predictions
    predictions = generate_mock_predictions()

    # Get top predictions
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_predictions = [
        {"disease": disease, "probability": prob}
        for disease, prob in sorted_preds[:5]
    ]

    # Generate co-occurrence matrix
    co_occurrence = generate_mock_co_occurrence(predictions)

    # Top disease
    top_disease = sorted_preds[0][0]
    top_disease_prob = sorted_preds[0][1]

    # Mock images (using sample from public folder)
    # Note: Frontend serves these from /public folder
    mock_result = {
        "predictions": predictions,
        "top_predictions": top_predictions,
        "co_occurrence": co_occurrence.tolist(),
        "disease_classes": config.DISEASE_CLASSES,
        "heatmap_overlay": "http://localhost:3000/xraygradcam.png",  # With Grad-CAM
        "original_image": "http://localhost:3000/xraygradcam.png",   # Same for demo
        "top_disease": top_disease,
        "top_disease_probability": top_disease_prob
    }

    return mock_result


# Predefined sample results for consistent demo
SAMPLE_RESULTS = [
    {
        "scenario": "Pneumonia + Infiltration",
        "primary": "Pneumonia",
        "secondary": "Infiltration",
        "detected": ["Pneumonia", "Infiltration"]
    },
    {
        "scenario": "Cardiomegaly + Effusion",
        "primary": "Cardiomegaly",
        "secondary": "Effusion",
        "detected": ["Cardiomegaly", "Effusion"]
    },
    {
        "scenario": "Atelectasis + Consolidation",
        "primary": "Atelectasis",
        "secondary": "Consolidation",
        "detected": ["Atelectasis", "Consolidation"]
    }
]


def get_sample_result(scenario_index: int = 0) -> Dict:
    """
    Get a predefined sample result for consistent demos
    Returns detected diseases without showing probabilities

    Args:
        scenario_index: Index of scenario (0-2)

    Returns:
        Mock analysis result with detected diseases
    """
    scenario_index = scenario_index % len(SAMPLE_RESULTS)
    scenario = SAMPLE_RESULTS[scenario_index]

    # Build simple detected/not detected for each disease
    predictions = {}
    detected_diseases = []

    for disease in config.DISEASE_CLASSES:
        if disease in scenario["detected"]:
            predictions[disease] = 1.0  # Detected
            detected_diseases.append({
                "disease": disease,
                "status": "detected",
                "is_primary": disease == scenario["primary"],
                "is_secondary": disease == scenario["secondary"]
            })
        else:
            predictions[disease] = 0.0  # Not detected

    # Generate co-occurrence with emphasis on detected diseases
    co_occurrence_probs = {d: 0.5 if d in scenario["detected"] else 0.1 for d in config.DISEASE_CLASSES}
    co_occurrence = generate_mock_co_occurrence(
        co_occurrence_probs,
        primary_disease=scenario["primary"],
        secondary_disease=scenario["secondary"]
    )

    return {
        "predictions": predictions,
        "detected_diseases": detected_diseases,
        "primary_disease": scenario["primary"],
        "secondary_disease": scenario["secondary"],
        "top_disease": scenario["primary"],  # For HeatmapViewer component
        "total_detected": len(scenario["detected"]),
        "co_occurrence": co_occurrence.tolist(),
        "disease_classes": config.DISEASE_CLASSES,
        "heatmap_overlay": "http://localhost:3000/xraygradcam.png",
        "original_image": "http://localhost:3000/xraygradcam.png",
        "show_probabilities": False  # Flag for frontend
    }
