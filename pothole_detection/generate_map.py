import pandas as pd
import folium

def generate_map(csv_file, output_map_file):
    # Read the pothole coordinates CSV file
    coords_df = pd.read_csv(csv_file)
    
    # Check if there are any coordinates in the CSV
    if coords_df.empty:
        print("⚠ No pothole coordinates found in the CSV file.")
        return

    # Initialize the map (using the first pothole's coordinates to center the map)
    first_row = coords_df.iloc[0]
    m = folium.Map(location=[first_row['latitude'], first_row['longitude']], zoom_start=16)

    # Add markers for each pothole in the CSV
    for _, row in coords_df.iterrows():
        pothole_image = row['image']
        latitude = row['latitude']
        longitude = row['longitude']
        
        # Add a marker for each pothole
        folium.Marker(
            location=[latitude, longitude],
            popup=f"Pothole: {pothole_image}",
            icon=folium.Icon(color='red')
        ).add_to(m)

    # Save the map to an HTML file
    m.save(output_map_file)
    print(f"🌍 Pothole map saved to {output_map_file}. Opening map...")
    
    # Open the generated map in the default web browser
    import webbrowser
    webbrowser.open(output_map_file)

# Example usage
if __name__ == "__main__":
    csv_file = "pothole_coordinates/pothole_coordinates.csv"  # Path to your CSV file
    output_map_file = "pothole_map.html"  # Output map file name
    generate_map(csv_file, output_map_file)
