# SymptoScan 
 
🐾 AI-based animal disease screening: predicts the most likely disease from an animal's basic info, vital signs, and symptoms.

Who it's for: veterinarians and animal owners in remote areas, to speed up preliminary screening and help reduce losses.

⚠️ SymptoScan is a screening aid and does not replace a professional veterinary diagnosis.

Data Real: Kaggle animal disease dataset (431 records).

Generated: AI-generated dataset (10,031 records), used only to augment training data (15 records per disease).
Pipeline

Cleaning: dropped Breed, removed duplicates, standardized symptoms into boolean features.
Label unification: 139 → 113 disease names.

Class filtering: kept diseases with at least 3 real records (47 diseases, 336 records), split 70/30 (stratified).

Baseline: a Dummy Classifier (most frequent class) as a lower-bound reference (5.9% accuracy).

Models: Logistic Regression, Random Forest, and XGBoost, trained with and without generated data, and validated with 5-fold cross-validation.

Final model: the best-performing model (Random Forest, Real + Generated).
