import streamlit as st
import os
from predict_pipeline import predict_image
from urllib.parse import urlparse, parse_qs
from render_csv import ImagePredictor
import streamlit as st
import base64
import pandas as pd

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def display_csv_black(data_frame):
    st.markdown('<style>table td { color: black !important; }</style>', unsafe_allow_html=True)
    st.dataframe(data_frame)

def get_binary_file_downloader_html(filename, text):
    with open(filename, 'rb') as file:
        data = file.read()
    base64_encoded = base64.b64encode(data).decode()
    return f'<a href="data:file/csv;base64,{base64_encoded}" download="{filename}">{text}</a>'

def change_font_color(element, text, font_color):
    st.markdown(f'<{element} style="color: {font_color};">{text}</{element}>', unsafe_allow_html=True)

class Prediction:
    def prediction(self):
        change_font_color('h1',"Predict the Image",'black')
        change_font_color('p','Run the AI Batch Inference with a Single Button','black')


        # Create an instance of the ImagePredictor class
        predictor = ImagePredictor(images_directory="predict_data/",
                                   model_path="/Artifacts/best.onnx",
                                   csv_file="Data_Files/render.csv")

        if st.button("RUN BATCH INFERENCE"):
            # Perform the inference and CSV generation
            predictor.perform_inference()
            df = load_csv("Data_Files/render.csv")
            change_font_color('h1',"Final Summary of Inference made for  Data in Database",'black')

            # Display CSV data in black
            display_csv_black(df)



            # Display a link to download the CSV file
            change_font_color('h1',"Download CSV File",'black')
            st.markdown(get_binary_file_downloader_html('Data_Files/render.csv', 'Download CSV'), unsafe_allow_html=True)

        st.sidebar.markdown('<h1 style="color: black;">Upload Image(s)</h1>', unsafe_allow_html=True)
        st.sidebar.markdown('<label for="file-uploader" style="color: black;">Choose image(s)...</label>',
                            unsafe_allow_html=True)
        uploaded_files = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"],
                                                  accept_multiple_files=True)

        if uploaded_files:
            # Create the predict_data directory if it doesn't exist
            os.makedirs("uploaded_images", exist_ok=True)

            # Save uploaded images to predict_data directory
            saved_image_paths = []
            for uploaded_file in uploaded_files:
                with open(os.path.join("uploaded_images", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.read())
                saved_image_paths.append(os.path.join("uploaded_images", uploaded_file.name))

            # Perform prediction on each image and visualize the results
            for image_path in saved_image_paths:
                st.subheader(f"Predictions for Image: {os.path.basename(image_path)}")

                # Perform prediction using the predict_image function
                plastic_count, url, saved_image_path, lat, longi = predict_image(image_path, "uploaded_images")
                st.write(f"Plastic Count: {plastic_count}")

                if url=="No Meta Data":
                    st.write("GeoTag URL not available.")


                elif url:
                    st.write(f"GeoTag URL: {url}")

                    # Parse the latitude and longitude from the URL or geolocation data
                    # Replace this with your method to extract latitude and longitude from the geolocation URL or data

                    # Display the map with a marker at the detected geolocation
                    st.map(latitude=lat, longitude=longi,
                           zoom=30)  # You can adjust the 'zoom' parameter as needed to set the initial map zoom level
                else:
                    st.write("GeoTag URL not available.")

                # Display the predicted image with bounding boxes
                st.image(saved_image_path, caption='Predicted Image', use_column_width=True)

                # Add a line break between images
                st.write("---")








