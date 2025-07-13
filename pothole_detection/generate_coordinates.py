import csv
import random

# Output CSV path
csv_path = r"C:\Users\umadevi\OneDrive\Desktop\Miniproject\pothole-detection\pothole_coordinates.csv"

# Labels pool (50% pothole, 50% no pothole approx.)
labels = ['pothole', 'no pothole']

# Generate 665 entries
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "latitude", "longitude", "label"])  # Updated header with 'label'

    for i in range(665):  # potholes0.png to potholes664.png
        image_name = f"potholes{i}.png"

        # Random coordinates near Hyderabad
        lat = round(17.385 + random.uniform(-0.01, 0.01), 6)
        lon = round(78.4867 + random.uniform(-0.01, 0.01), 6)

        # Random label (pothole or no pothole)
        label = random.choice(labels)

        writer.writerow([image_name, lat, lon, label])

print("✅ pothole_coordinates.csv file with labels generated successfully.")
