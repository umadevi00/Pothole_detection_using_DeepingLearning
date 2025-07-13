import os
import base64
import folium
import csv
import webbrowser
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TE5iDtuSU8YSdfPmmDSX"
)

# Paths
project_root = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(project_root, "test_images")
coordinates_path = os.path.join(project_root, "pothole_coordinates.csv")
ground_truth_path = os.path.join(project_root, "ground_truth.csv")
maps_folder = os.path.join(project_root, "pothole_maps")

os.makedirs(maps_folder, exist_ok=True)

# Load coordinates
coordinates = {}
if os.path.exists(coordinates_path):
    with open(coordinates_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            coordinates[row[0]] = {
                "latitude": row[1],
                "longitude": row[2]
            }

# Initialize map
pothole_map = folium.Map(location=[17.385044, 78.486671], zoom_start=12)

# Detection lists
y_true = []
y_scores = []

# Load ground truth labels
ground_truth = {}
if os.path.exists(ground_truth_path):
    with open(ground_truth_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_name, label = row[0], int(row[1])
            ground_truth[image_name] = label

# Process each image
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, image_file)
        print(f"\n🖼️ Processing: {image_file}")

        try:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")

            result = CLIENT.infer(encoded_image, model_id="pothole_detection-vuxge/1")
            predictions = result.get("predictions", [])

            if predictions:
                print("✅ Pothole(s) detected!")
                confidence_scores = [p['confidence'] for p in predictions]
                max_conf = max(confidence_scores)
                y_scores.append(max_conf)
            else:
                print("✅ No pothole detected.")
                max_conf = 0.0
                y_scores.append(max_conf)

            # Add true label
            if image_file in ground_truth:
                y_true.append(ground_truth[image_file])

            # Add to map
            if predictions and image_file in coordinates:
                lat = float(coordinates[image_file]["latitude"])
                lon = float(coordinates[image_file]["longitude"])
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Image: {image_file}\nConfidence: {max_conf * 100:.2f}%",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(pothole_map)

        except Exception as e:
            print(f"❌ Error processing {image_file}: {str(e)}")

# Save map
final_map_path = os.path.join(maps_folder, "all_potholes_map.html")
pothole_map.save(final_map_path)
print(f"\n✅ Final map saved at {final_map_path}")
webbrowser.open(f'file:///{final_map_path}')

# Plot ROC Curve
if len(y_true) >= 2:
    print("\n📈 Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Pothole Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Not enough labeled data for ROC curve.")

# Plot Confusion Matrix
if len(y_true) == len(y_scores):
    threshold = 0.5
    y_pred = [1 if score >= threshold else 0 for score in y_scores]

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # [Pothole, No Pothole]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pothole", "No Pothole"])

    print("\n📊 Confusion Matrix:")
    print(f"""
                Predicted
            | Pothole | No Pothole
Actual -----+---------+------------
Pothole     |   {cm[0][0]:<5}   |    {cm[0][1]:<5}
No Pothole  |   {cm[1][0]:<5}   |    {cm[1][1]:<5}
    """)
 
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Pothole Detection")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ y_true and y_scores length mismatch. Confusion Matrix skipped.")
