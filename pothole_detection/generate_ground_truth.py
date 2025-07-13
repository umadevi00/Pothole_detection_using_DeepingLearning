import os
import csv

test_images_dir = "test_images"
output_csv = "ground_truth.csv"

data = []

# Go through all files in test_images/
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        if "no" in filename.lower():
            label = 0
        else:
            label = 1
        data.append([filename, label])

# Write to CSV
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(data)

print(f"✅ ground_truth.csv generated with {len(data)} entries.")
