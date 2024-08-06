### Documentation for YOLO_VIDEO.PY

#### Overview
This code implements a video-based security system using the YOLO (You Only Look Once) object detection model for detecting weapons and a pre-trained machine learning model for violence detection. The system sends email alerts and plays a warning sound in case a weapon and potential violence are detected in the video stream.

#### Dependencies
- torch: PyTorch library for deep learning
- numpy: NumPy library for numerical operations
- cv2: OpenCV library for computer vision
- math: Python math library for mathematical operations
- ultralytics: YOLO is a part of Ultralytics
- smtplib: Simple Mail Transfer Protocol library for sending emails
- email.mime: MIME library for creating email messages
- pygame: Pygame library for sound handling
- tensorflow.keras: Keras library for machine learning
- tensorflow: TensorFlow library for machine learning
- time: Python time library for handling time-related operations

#### Setup
- Define email credentials (password, sender email, and recipient email).
- Set up the SMTP server for sending emails.
- Initialize Pygame mixer for sound.

#### Functions

##### `send_email(to_email, from_email, object_detected=1)`
- Sends an email alert with the specified subject and body.
- Parameters:
  - `to_email`: Email address of the recipient.
  - `from_email`: Email address of the sender.
  - `object_detected`: Number of detected objects (default is 1).

##### `stop_alert_sound()`
- Stops the currently playing alert sound.

##### `play_alert_sound()`
- Plays the predefined alert sound for a specified duration.

##### `video_detection(video_source)`
- Performs video detection using YOLO for weapon detection and a pre-trained model for violence detection.
- Parameters:
  - `video_source`: Path or source of the video stream.
- Loads YOLO and violence detection models.
- Processes each frame from the video stream:
  - Detects weapons using YOLO.
  - Performs violence detection using the pre-trained model.
  - Plays alert sound and sends an email if a weapon and potential violence are detected.
  - Displays annotated frames with detected objects.

#### Usage
- Specify the path to the video source.
- Run the code to start video-based security monitoring.



### Documentation for FLASKAPP.PY
#### Overview
This Flask application implements a video surveillance system that utilizes the YOLO (You Only Look Once) object detection model for real-time detection of objects, particularly focusing on security applications. The system allows users to upload a video file or stream video from a webcam or external cameras. Detected frames are displayed on the web interface.

#### Dependencies
- Flask: Web framework for Python
- Flask-WTF: Flask extension for integrating WTForms
- SQLite3: Lightweight database engine
- datetime: Module for working with dates and times
- wtforms: Library for form handling
- OpenCV: Open Source Computer Vision Library
- YOLO_Video: Custom module for YOLO-based video detection
- asyncio: Asynchronous I/O operations in Python

#### Flask App Setup
- The Flask app is initialized, and a secret key is set for security.
- Upload folder and file upload configurations are defined.

#### Form Handling
- A `FlaskForm` class, `UploadFileForm`, is defined using WTForms to handle file uploads.

#### Frame Generation Functions
- `generate_frames(path_x="")`: Generates frames from the video stream obtained from the specified path. Uses the `video_detection` function from the `YOLO_Video` module.
- `generate_frames_web(path_x)`: Generates frames from the webcam or external camera video stream using the `video_detection` function.


#### Video Surveillance Features
- Object detection is performed in real-time using the YOLO model.
- Detected frames are displayed on the web interface.
- The system supports video file upload and streaming from webcams or external cameras.



### Documentation for Face Recognition and Person tracking System

#### Overview
This Python script implements a face recognition and person tracking system using the Dlib library. The system detects faces in real-time, recognizes them based on pre-trained face features, and marks tracked in a SQLite database. 

#### Dependencies
- dlib: C++ toolkit for machine learning and computer vision
- numpy: NumPy library for numerical operations
- cv2: OpenCV library for computer vision
- pandas: Data manipulation and analysis library
- sqlite3: SQLite database engine
- datetime: Module for working with dates and times
- logging: Module for event logging

#### Dlib Setup
- Frontal face detector, shape predictor, and face recognition model are initialized using Dlib.

#### SQLite Database Setup
- Connection to the SQLite database is established, and a table named "attendance" is created for the current date.
- The table has columns: name (TEXT), time (TEXT), and date (DATE).

#### Face_Recognizer Class
- Manages face recognition and attendance tracking functionality.
- Utilizes the centroid tracker to track faces across frames.
- Calculates the Euclidean distance between face features for recognition.
- Updates and displays real-time information on the frame, including frame count, frames per second (FPS), and face count.

#### Attendance Tracking
- The `attendance` method inserts track records into the SQLite database.

#### Face Recognition Process
1. Faces are detected in each frame using Dlib's frontal face detector.
2. If the number of faces remains the same between frames, the centroid tracker is used to track faces.
3. If the number of faces changes, the script performs face recognition on the new faces.
4. The Euclidean distance is calculated between the face features of the current frame and known faces in the database.
5. If a match is found (distance < 0.4), the person's name is recognized, and attendance is marked.
6. Real-time information is displayed on the frame, and the process continues until the user presses 'q' to quit.

#### Usage
- Run the script to start the face recognition and person tracking system.
- Ensure that the required dependencies are installed and the pre-trained Dlib models are available.

#### Note
- Ensure that the attendance database (`attendance.db`) is in the same directory as the script.
- This script is designed to work with a webcam (device index 0). Modify the `cv2.VideoCapture` argument to use a different video source.
- Adjust the recognition threshold (`min(self.current_frame_face_X_e_distance_list) < 0.4`) based on your application's requirements.

#### Security Considerations
- Protect sensitive information such as the attendance database and the pre-trained face features CSV file.
- Implement additional security measures if deploying the system in a public or critical environment.

