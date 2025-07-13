import os
import base64
import folium
import cv2
import csv
import webbrowser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TE5iDtuSU8YSdfPmmDSX"
)

# Set project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Image and CSV paths
image_name = "no_potholes5.png"
image_path = os.path.join(project_root, "test_images", image_name)
coordinates_path = os.path.join(project_root, "pothole_coordinates.csv")

# Check image existence
if not os.path.exists(image_path):
    print(f"❌ Image {image_path} not found.")
    exit()
image = cv2.imread(image_path)
if image is None:
    print("❌ Failed to load image.")
    exit()

# Load label from CSV (label column expected)
actual_label = "no pothole"  # default
coordinates = {}
with open(coordinates_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        coordinates[row["image_name"]] = {
            "latitude": row.get("latitude", "0"),
            "longitude": row.get("longitude", "0"),
            "label": row.get("label", "no pothole").strip().lower()
        }
actual_label = coordinates.get(image_name, {}).get("label", "no pothole")

# Encode image for Roboflow
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")

try:
    # Run inference
    result = CLIENT.infer(encoded_image, model_id="pothole_detection-vuxge/1")
    predictions = result.get("predictions", [])
    predicted_label = "pothole" if predictions else "no pothole"

    print(f"\n🖼 Image: {image_name}")
    if predictions:
        print("✅ Pothole(s) detected!\n")
        pothole_map = folium.Map(location=[17.385044, 78.486671], zoom_start=12)

        for i, p in enumerate(predictions):
            confidence = p['confidence'] * 100
            status = "Accurate Detection" if confidence >= 80 else "Low Confidence"
            class_name = p['class']
            x, y = int(p['x']), int(p['y'])
            w, h = int(p['width']), int(p['height'])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            # ✅ Print detection info
            print(f"📌 Detection #{i+1}")
            print(f"   ➤ Class       : {class_name}")
            print(f"   ➤ Confidence  : {confidence:.2f}%")
            print(f"   ➤ Status      : {status}")
            print(f"   ➤ Bounding Box: (x={x1}, y={y1}, w={w}, h={h})\n")

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name} ({confidence:.1f}%)"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add marker to map
            lat = float(coordinates.get(image_name, {}).get("latitude", 0))
            lon = float(coordinates.get(image_name, {}).get("longitude", 0))
            if lat and lon:
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Image: {image_name}\nConfidence: {confidence:.2f}%\nStatus: {status}",
                    icon=folium.Icon(color='red')
                ).add_to(pothole_map)

        # Save and open map
        map_file = os.path.join(project_root, "detected_pothole_map.html")
        pothole_map.save(map_file)
        webbrowser.open(f'file:///{map_file}')
        print(f"✅ Map saved at {map_file}")

    else:
        print("✅ No pothole detected in this image.")

    # Save and display image
    result_path = os.path.join(project_root, "detection_result.png")
    cv2.imwrite(result_path, image)
    print(f"🖼 Detection image saved at {result_path}")

    # Show result image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title("Pothole Detection Result")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ✅ Confusion Matrix
    print(f"\n📊 Confusion Matrix")
    print(f"   ➤ Actual Label   : {actual_label}")
    print(f"   ➤ Predicted Label: {predicted_label}")

    labels = ["no pothole", "pothole"]
    cm = confusion_matrix([actual_label], [predicted_label], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Error during inference: {str(e)}")
