import cv2
import numpy as np
import random

def two_pass_labeling(binary_image):
    # Ensure binary image has values 0 and 1
    binary_image = (binary_image > 0).astype(np.uint8)

    # Step 1: First pass - initial labeling and equivalence table
    rows, cols = binary_image.shape
    labels = np.zeros_like(binary_image, dtype=np.int32)  # Matrix to store component labels
    label = 1  # Start labeling from 1
    equivalences = {}  # Dictionary to store equivalence information between labels

    # Iterate through the binary image
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:  # Process only foreground pixels
                # Get neighbors' labels (top and left)
                neighbors = []
                if i > 0 and labels[i - 1, j] > 0:  # Top neighbor
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:  # Left neighbor
                    neighbors.append(labels[i, j - 1])

                if not neighbors:
                    # No neighbors - assign a new label
                    labels[i, j] = label
                    equivalences[label] = {label}  # Initialize equivalence set
                    label += 1
                else:
                    # Neighbors exist - assign the smallest label
                    min_label = min(neighbors)
                    labels[i, j] = min_label

                    # Update equivalence sets
                    for neighbor_label in neighbors:
                        equivalences[min_label].update(equivalences[neighbor_label])
                        equivalences[neighbor_label].update(equivalences[min_label])

    # Step 2: Flatten equivalence table to resolve all equivalences
    for key, value in equivalences.items():
        for v in value:
            equivalences[v] = value

    # Assign final labels based on equivalence resolution
    final_labels = {}  # Map each root label to a new unique label
    new_label = 1
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] > 0:  # Only process labeled pixels
                root_label = min(equivalences[labels[i, j]])  # Get root label
                if root_label not in final_labels:
                    # Assign a new unique label to this root label
                    final_labels[root_label] = new_label
                    new_label += 1
                labels[i, j] = final_labels[root_label]  # Update the label in the image

    # Step 3: Generate color-labeled image for visualization
    label_count = new_label - 1  # Total number of connected components
    color_labeled_image = np.zeros((rows, cols, 3), dtype=np.uint8)  # Initialize a blank color image
    colors = {
        label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for label in range(1, label_count + 1)
    }  # Assign random colors to labels

    for i in range(rows):
        for j in range(cols):
            if labels[i, j] > 0:  # Apply color to labeled pixels
                color_labeled_image[i, j] = colors[labels[i, j]]

    return color_labeled_image

# Example usage
if __name__ == "__main__":
    # Read binary image
    binary_image = cv2.imread("Sample Image 1.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (values 0 or 255)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Apply two-pass algorithm to label connected components
    color_labeled_image = two_pass_labeling(binary_image)

    # Save the resulting labeled image
    cv2.imwrite("labeled_output.png", color_labeled_image)

    # Display the result
    cv2.imshow("Labeled Components", color_labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
