import configparser
import cv2
import time

def detect_faces(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

def process_frame(frame, blur_enabled, blur_color):
    # Detect faces in the frame
    faces = detect_faces(frame)

    # Draw rectangles around the detected faces and apply blur if enabled
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face with the selected color
        cv2.rectangle(frame, (x, y), (x+w, y+h), blur_color, 2)

        if blur_enabled:
            # Apply blur to the face region
            blurred_face = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)

            # Replace the face region with the blurred version
            frame[y:y+h, x:x+w] = blurred_face

    # Display the FPS on the frame
    fps = round(1 / (time.time() - start_time), 2)
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the processed frame in a full-screen window
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Frame', frame)

def process_video_file(video_path, blur_enabled, blur_color):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame
        process_frame(frame, blur_enabled, blur_color)

        # Wait for a key press
        key = cv2.waitKey(1)

        # Exit the program if the Q key is pressed
        if key == ord('Q') or key == ord('q'):
            break

        # Toggle the blur if the B key is pressed
        if key == ord('B') or key == ord('b'):
            blur_enabled = not blur_enabled

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    video_file_path = config.get('Config', 'video_file_path')
    blur_enabled = config.getboolean('Config', 'blur_enabled')
    blur_color_r = config.getint('Config', 'blur_color_r')
    blur_color_g = config.getint('Config', 'blur_color_g')
    blur_color_b = config.getint('Config', 'blur_color_b')

    return {
        'video_file_path': video_file_path,
        'blur_enabled': blur_enabled,
        'blur_color': (blur_color_r, blur_color_g, blur_color_b)
    }

config_path = r"C:\Users\anzhe\PycharmProjects\pythonProject8\config.ini"
config = load_config(config_path)

video_file_path = config["video_file_path"]
blur_enabled = config["blur_enabled"]
blur_color = config["blur_color"]

start_time = time.time()

if video_file_path:
    process_video_file(video_file_path, blur_enabled, blur_color)