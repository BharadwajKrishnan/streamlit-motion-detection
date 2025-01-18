import streamlit as st
import cv2
import tempfile
import os
import numpy as np

# Function for motion detection and video annotation
def motion_detection(video_path, output_path):
    # Create video reader object
    video_obj = cv2.VideoCapture(video_path)

    # Get fps, frame width, and height of the video
    video_fps = int(video_obj.get(cv2.CAP_PROP_FPS))
    video_width = int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (video_width, video_height))

    # Set kernel size for erosion
    kernel_size = (5, 5)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    max_contours = 3  # Annotate only the largest 3 contours

    # Create background subtractor object
    bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

    while True:
        # Read a frame from the video
        has_frame, frame = video_obj.read()

        if not has_frame:
            break
        else:
            frame_erode = frame.copy()

        # Create foreground mask and perform erosion
        fg_mask = bg_sub.apply(frame)
        fg_mask_erode = cv2.erode(fg_mask, np.ones(kernel_size, np.uint8))

        # Find motion areas
        motion_area_erode = cv2.findNonZero(fg_mask_erode)
        if motion_area_erode is not None:
            # Get the bounding rectangle for the motion area
            xe, ye, we, he = cv2.boundingRect(motion_area_erode)

            # Draw the bounding rectangle on the frame
            cv2.rectangle(frame_erode, (xe, ye), (xe + we, ye + he), red)

        # Find contours on the eroded frame
        contours, _ = cv2.findContours(fg_mask_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours are found
        if len((contours)) > 0:
            # Sort contours by area and annotate the largest ones
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]

            for cnt_id in range(min(max_contours, len(contours))):
                xc, yc, wc, hc = cv2.boundingRect(contours_sorted[cnt_id])

                if cnt_id == 0:
                    x1 = xc
                    y1 = yc
                    w1 = wc
                    h1 = hc
                else:
                    x1 = min(x1, xc)
                    y1 = min(y1, yc)
                    w1 = max(w1, wc)
                    h1 = max(h1, hc)

            # Draw the bounding rectangle
            cv2.rectangle(frame_erode, (x1, y1), (x1 + w1, y1 + h1), blue)

        # Write the annotated frame
        video_writer.write(frame_erode)

    # Release resources
    video_writer.release()
    video_obj.release()

# Streamlit UI
st.title("Motion Detection with OpenCV")

# File uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())
        input_video_path = temp_input.name

    # Temporary file for output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = temp_output.name

    # Process the video
    motion_detection(input_video_path, output_video_path)

    # Create two columns for input and output video display
    columns = st.columns(1)

    # Display input video
    columns[0].header("Input Video")
    columns[0].video(input_video_path)

    # Provide a download link for the processed video
    with open(output_video_path, "rb") as file:
        processed_video_bytes = file.read()
        st.download_button(
            label="Download Processed Video",
            data=processed_video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
