# import torch
# import numpy as np
# import cv2
# import math
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# import pygame
# import numpy as np
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# import time

# password = "kuec cerx zeyx qcjm"
# from_email = "mrsachinchaurasiya@gmail.com"
# to_email = "sachinchaurasiya69@gmail.com"

# server = smtplib.SMTP("smtp.gmail.com: 587")
# server.starttls()
# server.login(from_email, password)

# # Initialize pygame mixer for sound
# pygame.mixer.init()

# # Load beep sound for violence detection
# beep_sound = pygame.mixer.Sound(
#     "../rajshtan-police-hackathon/VIOLANCE ALERT.mp3"
# )


# def send_email(to_email, from_email, object_detected=1):
#     message = MIMEMultipart()
#     message["From"] = from_email
#     message["To"] = to_email
#     message["Subject"] = "Security Alert"
#     message_body = f"ALERT - {object_detected} There is some quarrel detected from the cell, and it's an emergency to look into this matter, as the fight looks quite serious."

#     message.attach(MIMEText(message_body, "plain"))
#     server.sendmail(from_email, to_email, message.as_string())


# alert_duration = 5  # Duration of the alert sound in seconds


# def stop_alert_sound():
#     pygame.mixer.stop()


# def play_alert_sound():
#     beep_sound.play()
#     time.sleep(
#         alert_duration
#     )  # Introduce a delay to play the alert sound for 5 seconds
#     stop_alert_sound()


# def video_detection(video_source):
#     cap = cv2.VideoCapture(video_source)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     # Load YOLO model
#     model1 = YOLO("../YOLO-Weights/best.pt")
#     model1.model.to("cuda")
#     model1.model.eval()

#     # Load violence detection model

#     # Load violence detection model
#     violence_model = tf.keras.models.load_model(
#         r"../rajshtan-police-hackathon/modelnew(1).h5"
#     )

#     # Check if GPU is available for TensorFlow
#     if tf.test.is_gpu_available():
#         print("GPU is available for TensorFlow")
#         violence_model = tf.keras.models.clone_model(violence_model)
#         violence_model.compile(
#             optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
#         )
#         violence_model = violence_model.to(device="cuda")

#     classNames_model1 = ["Granade", "Gun", "Knife", "Pistol", "handgun", "rifle"]

#     alert_active = False  # Flag to track if alert sound is currently active

#     while True:
#         success, img = cap.read()
#         results = model1(img, stream=True)
#         weapon_detected = False
#         img_array = cv2.resize(
#             img, (128, 128)
#         )  # Resize frame to match model's expected sizing
#         img_array = (
#             np.expand_dims(img_array, axis=0) / 255.0
#         )  # Normalize pixel values to be between 0 and 1

#         # Make predictions
#         predictions = violence_model.predict(img_array)

#         # Assuming it's a binary classification (violence or not violence)
#         if predictions[0][0] > 0.3:
#             print("Violence detected.")

#             # Play beep sound
#             play_alert_sound()

#         else:
#             print("No violence detected.")
            

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 conf = math.ceil((box.conf[0] * 100)) / 100

#                 if conf >= 0.7:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     print(x1, y1, x2, y2)
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#                     cls = int(box.cls[0])
#                     class_name = classNames_model1[cls]
#                     label = f"{class_name}{conf}"
#                     t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                     print(t_size)
#                     c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                     cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
#                     cv2.putText(
#                         img,
#                         label,
#                         (x1, y1 - 2),
#                         0,
#                         1,
#                         [255, 255, 255],
#                         thickness=1,
#                         lineType=cv2.LINE_AA,
#                     )

#                     # Set flag to indicate weapon detection
#                     weapon_detected = True

#         # Perform violence detection using the second model

#         # Play or stop alert sound based on weapon and violence detection
#                 if weapon_detected and not alert_active:
#                       play_alert_sound()
#                       alert_active = True
#                       send_email(to_email, from_email)
#                 elif not weapon_detected and alert_active:
#                     alert_active = False

#         yield img
import torch
print(torch.cuda.is_available())
