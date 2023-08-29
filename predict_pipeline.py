import argparse
import os
import torch

from get_description import image_coordinates
from PIL import Image, ImageDraw, ImageFont

def predict_image(image_path, output_dir):
    # Load the trained model
    model = torch.hub.load('Utility_dir',
                           'custom', 'Artifacts/best.onnx', source='local')

    # Load and preprocess the image
    img = Image.open(image_path).resize((640, 640))

    # Perform object detection on the image
    results = model(img, size=640)

    # Get the predicted labels and their counts
    pred_labels = results.pandas().xyxy[0]['name']
    print(pred_labels)
    plastic_count = (pred_labels == 'plastic').sum()

    # Filter the results to only include "plastic" class
    plastic_boxes = results.pred[0][pred_labels == 'plastic']
    # get the plastic confidense score
    df = results.pandas().xyxy[0]
    print(df)

    plastic_scores = list(df[df['name'] == 'plastic']['confidence'])
    print(plastic_scores)

    # Draw bounding boxes and labels for the filtered "plastic" class
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 14)  # Specify the font and size for the label

    for box, score in zip(plastic_boxes, plastic_scores):
        # Extract coordinates
        x1, y1, x2, y2 = box[0:4]
        coordinates = [(x1, y1), (x2, y2)]

        # Draw rectangle
        draw.rectangle(coordinates, outline=(0, 255, 0), width=2)

        # Add label "Plastic" with confidence score
        label = f"Plastic: {score:.2f}"
        label_font = ImageFont.truetype("arial.ttf", 12)  # Specify font and size for the label text
        label_size = label_font.getsize(label)
        draw.rectangle([(x1, y1), (x1 + label_size[0] + 4, y1 + label_size[1] + 4)], fill=(0, 255, 0))
        draw.text((x1 + 2, y1 + 2), label, fill=(0, 0, 0), font=label_font)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)




    # Save the predicted image with bounding boxes
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    img.save(save_path)

    # Get the GeoTag URL

    url,lat,longi = image_coordinates(image_path)
    print(url,lat,longi)

    return plastic_count, url, save_path,lat,longi

def main(input_path, output_dir):
    # Check if the input path is a file
    if os.path.isfile(input_path):
        plastic_count, url, saved_image_path,lat,longi = predict_image(input_path, output_dir)
        print(f"The prediction for image {input_path}:")
        print(f"Plastic Count: {plastic_count}")
        print(f"GeoTag URL: {url}")
        print(f"Saved Predicted Image: {saved_image_path}")
    # Check if the input path is a directory
    elif os.path.isdir(input_path):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over the images in the directory
        for image_file in os.listdir(input_path):
            # Get the image path
            image_path = os.path.join(input_path, image_file)

            # Predict the image and get the counts, URLs, and saved image path
            plastic_count, url, saved_image_path,lat,longi = predict_image(image_path, output_dir)

            # Print the prediction results
            print(f"The prediction for image {image_file}:")
            print(f"Plastic Count: {plastic_count}")
            print(f"GeoTag URL: {url}")
            print(f"Saved Predicted Image: {saved_image_path}")
            print()

    else:
        print("Invalid input path. Please provide a valid file or directory path.")


