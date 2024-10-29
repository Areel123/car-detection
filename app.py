import cv2
import numpy as np
import tempfile
import streamlit as st

st.title("Vehicle Counting Web App")
st.write("Upload a video file, and the app will count the vehicles that pass a defined line.")

# Parameters
min_width_react = 80
min_height_react = 80
count_line_position = 550
offset = 6

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)
    algo = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    car_count = 0
    detections = []  # To track the center points of vehicles that have passed the line

    stframe = st.empty()
    car_counter_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        img_sub = algo.apply(blur)
        dilated = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        # Track vehicles
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h

            # Filter out non-vehicle contours
            if (w >= min_width_react and h >= min_height_react and 
                contour_area > 500 and 
                (1.2 < aspect_ratio < 4.0 or 1.0 < aspect_ratio < 2.0)):
                center_point = (x + w // 2, y + h // 2)

                # Only count if the center point has crossed the line
                if (count_line_position - offset) < center_point[1] < (count_line_position + offset):
                    if center_point not in detections:  # Ensure we only count once
                        car_count += 1
                        detections.append(center_point)

                # Draw rectangle around detected vehicles
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display total count of vehicles
        cv2.putText(frame, f"Vehicle Count: {car_count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        car_counter_display.write(f"Total Vehicles Counted: {car_count}")

    cap.release()
    st.write(f"Video processing complete. Total vehicles counted: {car_count}")
