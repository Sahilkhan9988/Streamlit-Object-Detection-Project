import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import torch
import imutils
from ultralytics import YOLO



# Function to perform object detection on an image
def predict_image(model, img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    result_images = []
    for r in results:
        im_array = r.plot()
        result_images.append(im_array)

    return result_images

# Function to perform object detection on a video frame
def predict_frame(model, frame, conf_threshold, iou_threshold):
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    result_images = []
    for r in results:
        im_array = r.plot()
        result_images.append(im_array)

    return result_images

# Function to start object tracking
def start_object_tracking(frame, boxes):
    trackers = []
    for box in boxes:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(box))
        trackers.append(tracker)
    return trackers

# Function to perform real-time object detection and tracking
def real_time_detection_and_tracking(model, confidence_threshold, iou_threshold):
    cap = cv2.VideoCapture(0)
    trackers = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)

        object_detection_result_images = predict_frame(model, frame, confidence_threshold, iou_threshold)
        st.image(object_detection_result_images[0], caption='Detected Objects', use_column_width=True)

        if len(trackers) == 0:
            boxes = []  
            for result in object_detection_result_images:
                pass  
            trackers = start_object_tracking(frame, boxes)
        else:
            for tracker in trackers:
                success, box = tracker.update(frame)
                if success:




                    x, y, w, h = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    trackers.remove(tracker)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption='Real-Time Object Detection', use_column_width=True)

        if st.button("Stop Camera"):
            break

    cap.release()
    st.success("Camera Closed Successfully!")

def home():
    st.title("Zeeshan Khan Sahil")
    image_path = "istockphoto-1364317541-612x612.jpg"
    st.image(image_path, caption='Your Profile Picture', use_column_width=True, output_format='JPEG')


# Main function for the project
def Project():
    st.sidebar.title("Detection")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
    iou_threshold = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.45)

    # Model selection dropdown for object detection
    object_detection_model_options = ["pt", "seg", "pose", "obb", "cls"]
    selected_object_detection_model = st.sidebar.selectbox("Select Object Detection Model", object_detection_model_options)

    # Load the selected object detection model
    if selected_object_detection_model:
        if selected_object_detection_model == "pt":
            object_detection_model = YOLO("yolov8m.pt")
        elif selected_object_detection_model == "seg":
            object_detection_model = YOLO("yolov8m-seg.pt")
        elif selected_object_detection_model == "pose":
            object_detection_model = YOLO("yolov8m-pose.pt")
        elif selected_object_detection_model == "obb":
            object_detection_model = YOLO("yolov8m-obb.pt")
        elif selected_object_detection_model == "cls":
            object_detection_model = YOLO("yolov8m-cls.pt")

    # Model selection dropdown for instance segmentation
    instance_segmentation_model_options = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
    selected_instance_segmentation_model = st.sidebar.selectbox("Select Instance Segmentation Model", instance_segmentation_model_options, key="instance_segmentation_model_select")

    # Load the selected instance segmentation model
    if selected_instance_segmentation_model:
        instance_segmentation_model = torch.hub.load('ultralytics/yolov5', selected_instance_segmentation_model)

    st.title("Object Detection and Instance Segmentation")
    image_path = "compvision_tasks.png"
    st.image(image_path, caption='Your Profile Picture', use_column_width=True, output_format='JPEG')

    st.write("""The script enables users to upload images and videos and
    detect objects within them using object detection and instance segmentation models.""")

    uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4"])

    start_camera_button_key = "start_camera_button"
    stop_camera_button_key = "stop_camera_button"  # Base key for stop camera button
    
  
    # Add a counter variable to generate unique keys for the "Stop Camera" button
    stop_camera_button_counter = 0

    if st.button("Open Camera", key=start_camera_button_key):
        cap = cv2.VideoCapture(0)  # Access the first camera (0) of the system
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            object_detection_result_images = predict_frame(object_detection_model, frame, confidence_threshold, iou_threshold)
            st.image(object_detection_result_images[0], caption='Detected Objects', use_column_width=True)

            instance_segmentation_results = instance_segmentation_model(frame)
            instance_segmentation_result_images = instance_segmentation_results.render()
            st.image(instance_segmentation_result_images[0], caption='Instance Segmentation', use_column_width=True)

            # Increment the counter to generate a unique key for each button
            stop_camera_button_counter += 1

            if st.button("Stop Camera", key=f"{stop_camera_button_key}_{stop_camera_button_counter}"):
                break

        cap.release()
        st.success("Camera Closed Successfully!")

    
    # Add code for handling uploaded files similar to the existing code
    if uploaded_file is not None:
        # Handle both image and video uploads
        if uploaded_file.type.startswith('image') and selected_object_detection_model:
            # Handle image upload
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            img_array = np.array(image)

            object_detection_result_images = predict_image(object_detection_model, img_array, confidence_threshold, iou_threshold)
            st.subheader("Detected Objects")
            for idx, result_image in enumerate(object_detection_result_images, start=1):
                st.image(result_image, caption=f'Detected Objects {idx}', use_column_width=True)

            instance_segmentation_results = instance_segmentation_model(img_array)
            instance_segmentation_result_images = instance_segmentation_results.render()
            st.subheader("Instance Segmentation")
            for idx, result_image in enumerate(instance_segmentation_result_images, start=1):
                st.image(result_image, caption=f'Instance Segmentation {idx}', use_column_width=True)
        elif uploaded_file.type.startswith('video') and selected_object_detection_model:
            # Handle video upload
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            cap = cv2.VideoCapture(temp_file.name)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                object_detection_result_images = predict_frame(object_detection_model, frame, confidence_threshold, iou_threshold)
                st.image(object_detection_result_images[0], caption=f'Detected Objects in Frame {frame_idx}', use_column_width=True)

                instance_segmentation_results = instance_segmentation_model(frame)
                instance_segmentation_result_images = instance_segmentation_results.render()
                st.subheader("Instance Segmentation")
                for idx, result_image in enumerate(instance_segmentation_result_images, start=1):
                    st.image(result_image, caption=f'Instance Segmentation {idx}', use_column_width=True)

                frame_idx += 1

                # Display video
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, caption='Video', use_column_width=True)

            cap.release()

        # Add button to save detected images
        if st.button("Save Detected Images"):
            for idx, result_image in enumerate(object_detection_result_images, start=1):
                img_to_save = Image.fromarray(result_image)
                img_to_save.save(f"detected_image_{idx}.jpg")
            st.success("Detected images saved successfully!")
    else:
        st.info("Please upload an image or video.")

def About():
    st.title(" Import Libraries")
    st.write("""Importing Libraries: This section imports necessary libraries for the project, 
             including Streamlit, OpenCV (cv2), NumPy, PIL (Image module), os, tempfile, torch, 
             and imutils. It also imports the YOLO model from the Ultralytics library.
            """)

    st.title(" Object Detection Functions")
    st.write("""predict_image: Performs object detection on a single image.
            predict_frame: Performs object detection on a single frame of a video.
            start_object_tracking: Initializes object tracking using the specified boxes.
            real_time_detection_and_tracking: Performs real-time object detection and tracking using the webcam feed.""")

    
    st.title("Streamlit App Functions")
    st.write("""home: Displays the home page of the Streamlit app.
            Project: Main function displaying the project page with options for object detection 
             and instance segmentation.
            About: Displays information about the project.""")

    st.title("Integration of Object Detection with Streamlit")
    st.write("""
            Users can select object detection and instance segmentation models from the sidebar.
            They can upload images/videos or use real-time webcam feed for object detection.
            Detected objects are displayed with bounding boxes and labels in real-time.
            Instance segmentation results are also displayed if a model is selected.
            Users can save detected images.""")

    st.title("Sidebar Navigation")
    st.write("""The sidebar allows users to navigate between different pages of the app (Home, Project, About)""")

    st.title("Additional Features")
    st.write("""Model selection dropdowns for both object detection and instance segmentation.
            Slider widgets for adjusting confidence and IoU thresholds.
            Option to save detected images.
            Option to open the webcam for real-time object detection.""")

    st.title("Possible Improvements")
    st.write(""""
            Error handling for graceful handling of exceptions.
            UI/UX enhancements for better user experience.
            Adding more model options and flexibility for users.
            Better documentation for code understanding and maintenance.""")


    st.title("Comments")
    comment_name = st.text_input("Your Name:", "")
    comment_text = st.text_area("Add your comment here:", "")
    if st.button("Save Comment"):
        if comment_name and comment_text:
            with open("comments.txt", "a") as f:
                f.write(f"{comment_name}: {comment_text}\n")
            st.success("Comment saved successfully!")
        else:
            st.error("Please provide both your name and your comment.")

    # Show existing comments with delete buttons
    if os.path.exists("comments.txt"):
        with open("comments.txt", "r") as f:
            comments = f.readlines()
            if comments:
                st.title("Previous Comments")
                for idx, c in enumerate(comments, start=1):
                    try:
                        name, comment = c.split(":", 1)  # Split only once
                        if st.button(f"Delete Comment {idx}"):
                            # Check if the commenter's name matches the input name
                            if name.strip() == comment_name.strip():
                                comments.pop(idx-1)  # Remove the comment from the list
                                # Rewrite the comments file excluding the deleted comment
                                with open("comments.txt", "w") as new_f:
                                    new_f.writelines(comments)
                                st.success("Comment deleted successfully!")
                                break
                            else:
                                st.error("You can only delete your own comments!")
                        st.write(f"{idx}. {name.strip()}: {comment.strip()}")
                    except ValueError:
                        # Handle the case where the comment cannot be split into name and text
                        st.error(f"Error processing comment {idx}: {c}")


def show_sidebar():
    st.sidebar.title("Data Sciecetist")
    selected_page = st.sidebar.radio("Go to", ["Home", "Project", "About"])

    # Based on the user's selection, display the corresponding page
    if selected_page == "Home":
        home()
    elif selected_page == "Project":
        Project()
    elif selected_page == "About":
        About()

show_sidebar()


