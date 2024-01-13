from src.data_loader import load_and_prepare_data
from src.medium_tree_classifier import train_medium_tree_classifier, validate_medium_tree_classifier
from src.discriminant_classifier import train_discriminant_classifier, validate_discriminant_classifier
from src.coarse_tree_classifier import train_coarse_tree_classifier, validate_coarse_tree_classifier

# Load and prepare data
train_data, validation_data = load_and_prepare_data()

# Train and validate medium tree classifier
trained_medium_tree_classifier = train_medium_tree_classifier(train_data)
accuracy_medium_tree = validate_medium_tree_classifier(trained_medium_tree_classifier, validation_data)
print(f"Medium Tree Classifier Validation Accuracy: {accuracy_medium_tree:.4f}")

# Train and validate discriminant classifier
trained_discriminant_classifier = train_discriminant_classifier(train_data)
accuracy_discriminant = validate_discriminant_classifier(trained_discriminant_classifier, validation_data)
print(f"Discriminant Classifier Validation Accuracy: {accuracy_discriminant:.4f}")

# Train and validate coarse tree classifier
trained_coarse_tree_classifier = train_coarse_tree_classifier(train_data)
accuracy_coarse_tree = validate_coarse_tree_classifier(trained_coarse_tree_classifier, validation_data)
print(f"Coarse Tree Classifier Validation Accuracy: {accuracy_coarse_tree:.4f}")
