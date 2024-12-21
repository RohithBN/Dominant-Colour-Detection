import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

def determine_optimal_clusters(image, max_clusters=10, sample_size=10000):
    """Determine optimal number of clusters using the elbow method."""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Downsample pixels for faster processing
    pixels = image_rgb.reshape(-1, 3)
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Calculate distortions for different k values
    distortions = []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100, n_init=1)
        kmeans.fit(pixels)
        distortions.append(kmeans.inertia_)

    # Calculate the rate of change in distortion
    deltas = np.diff(distortions)
    k_values_for_rate = list(cluster_range)[2:]
    delta_ratios = np.abs(deltas[1:] / deltas[:-1])

    # Find elbow point
    elbow_point = np.argmin(delta_ratios) + 2

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot distortion curve
    ax[0].plot(list(cluster_range), distortions, 'bo-')
    ax[0].plot(elbow_point, distortions[elbow_point - 1], 'ro', markersize=10,
                label=f'Elbow Point (k={elbow_point})')
    ax[0].set_title('Elbow Method for Optimal k')
    ax[0].set_xlabel('Number of Clusters (k)')
    ax[0].set_ylabel('Distortion (Inertia)')
    ax[0].grid(True)
    ax[0].legend()

    # Plot rate of change
    ax[1].plot(k_values_for_rate[:len(delta_ratios)], delta_ratios, 'go-')
    ax[1].set_title('Rate of Change in Distortion')
    ax[1].set_xlabel('Number of Clusters (k)')
    ax[1].set_ylabel('Rate of Change')
    ax[1].grid(True)

    plt.tight_layout()
    return elbow_point, fig

def analyze_color_variations(image, num_colors, brightness_levels=5, sample_size=50000):
    """
    Analyze dominant colors and their brightness variations in the image.
    Returns color palettes for each dominant color.
    """
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better brightness analysis
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Downsample pixels for faster processing
    pixels_rgb = image_rgb.reshape(-1, 3)
    pixels_hsv = image_hsv.reshape(-1, 3)
    
    if len(pixels_rgb) > sample_size:
        indices = np.random.choice(len(pixels_rgb), sample_size, replace=False)
        sample_pixels = pixels_rgb[indices]
    else:
        sample_pixels = pixels_rgb

    # Perform K-means clustering on sampled pixels
    kmeans = KMeans(n_clusters=num_colors, random_state=42, max_iter=100, n_init=1)
    kmeans.fit(sample_pixels)
    
    # Get labels for all pixels
    labels = kmeans.predict(pixels_rgb)
    centers = kmeans.cluster_centers_

    # Initialize color palettes storage
    color_palettes = []

    # Process each dominant color
    for i in range(num_colors):
        # Get pixels belonging to this cluster
        cluster_mask = labels == i
        cluster_pixels_rgb = pixels_rgb[cluster_mask]
        cluster_pixels_hsv = pixels_hsv[cluster_mask]
        
        # Get the dominant color
        dominant_color = centers[i].astype(int)
        
        # Calculate percentage
        percentage = (np.sum(cluster_mask) / len(labels)) * 100
        
        # Sort cluster pixels by brightness (V in HSV)
        brightness_values = cluster_pixels_hsv[:, 2]
        sorted_indices = np.argsort(brightness_values)
        sorted_pixels = cluster_pixels_rgb[sorted_indices]
        
        # Create brightness variations
        variations = []
        chunk_size = len(sorted_pixels) // brightness_levels
        
        for j in range(brightness_levels):
            start_idx = j * chunk_size
            end_idx = (j + 1) * chunk_size if j < brightness_levels - 1 else len(sorted_pixels)
            if start_idx < end_idx:
                chunk_pixels = sorted_pixels[start_idx:end_idx]
                variation_color = np.mean(chunk_pixels, axis=0).astype(int)
                variations.append({
                    'rgb_color': tuple(variation_color),
                    'hex_color': '#{:02x}{:02x}{:02x}'.format(*variation_color)
                })
        
        # Store palette information
        color_palettes.append({
            'dominant_color': {
                'rgb_color': tuple(dominant_color),
                'hex_color': '#{:02x}{:02x}{:02x}'.format(*dominant_color)
            },
            'percentage': round(percentage, 2),
            'variations': variations
        })
    
    # Sort palettes by percentage
    color_palettes.sort(key=lambda x: x['percentage'], reverse=True)
    
    return color_palettes

def main():
    st.title("Image Color Analysis Tool")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Display dominant colors preview
        max_clusters = st.slider("Select max clusters for elbow method", 2, 15, 10)

        # Determine optimal clusters
        st.subheader("Determine Optimal Number of Clusters")
        optimal_k, elbow_fig = determine_optimal_clusters(image, max_clusters)
        st.pyplot(elbow_fig)

        st.write(f"Optimal number of clusters: {optimal_k}")

        # Analyze color variations
        st.subheader("Analyze Color Variations")
        brightness_levels = st.slider("Select number of brightness levels", 3, 10, 5)
        palettes = analyze_color_variations(image, num_colors=optimal_k, brightness_levels=brightness_levels)

        # Create a color preview for dominant colors
        st.subheader("Dominant Colors Preview")
        cols = st.columns(len(palettes))
        for i, (palette, col) in enumerate(zip(palettes, cols)):
            col.markdown(f"**Color {i+1}**")
            col.markdown(
                f"<div style='background-color:{palette['dominant_color']['hex_color']};height:100px;width:100%;'></div>",
                unsafe_allow_html=True
            )
            col.write(f"{palette['percentage']}%")

        # Display detailed results
        st.subheader("Color Variations")
        for i, palette in enumerate(palettes, 1):
            st.write(f"**Dominant Color {i} ({palette['percentage']}%):**")
            st.write(f"RGB: {palette['dominant_color']['rgb_color']}, Hex: {palette['dominant_color']['hex_color']}")

            col1, col2, col3 = st.columns(3)
            for j, variation in enumerate(palette['variations']):
                col = [col1, col2, col3][j % 3]
                col.markdown(f"Brightness Variation {j+1}: {variation['hex_color']}")
                col.markdown(f"<div style='background-color:{variation['hex_color']};height:50px;width:100%;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()