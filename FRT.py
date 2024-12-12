from ultralytics import YOLO
import cv2
import math
import torch
import smtplib
import pygame
import requests
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Function to send an alert to Directus
def send_to_directus(image_path, camera_no):
    url = "https://your-directus-instance.com/items/alerts"  # Replace with your Directus API endpoint
    headers = {
        "Authorization": "Bearer YOUR_ACCESS_TOKEN"  # Replace with your Directus access token
    }

    files = {
        'image': open(image_path, 'rb')  # Open the saved image file
    }
    data = {
        'camera_no': camera_no
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        print("Directus Alert Sent:", response.json())
    except Exception as e:
        print(f"Failed to send data to Directus: {e}")
    finally:
        files['image'].close()

# Function to send email alert
def send_email_alert():
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "mrsachinchaurasiya@gmail.com"
    sender_password = "uqoo ngyq beko puys"  # Secure this value
    receiver_email = "Sachinchaurasiya69@gmail.com"

    # Create the email content
    subject = "Weapon Detected Alert"
    body = "A weapon has been detected in the video stream."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

# Function to play sound alert
def play_sound_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("Alert_sound.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        print("Sound alert played successfully.")
    except Exception as e:
        print(f"Failed to play sound alert: {e}")

# Video detection function
def video_detection(path_x, camera_no):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    model = YOLO("../YOLO-Weights/Weapon.pt")  # Loading the model
    device = torch.device('cuda')
    model.to(device)

    classNames = ['Rifle', 'Hand Gun', 'Knife', 'Weapon', 'Test', 'Detecting', 'Gun']
    weapon_detected = False

    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                color = (0, 255, 0)

                if conf > 0.80:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    if class_name.lower() in ['rifle', 'hand gun', 'knife', 'weapon', 'gun']:
                        weapon_detected = True
                        if weapon_detected:
                            # Save the detected frame as an image
                            image_path = f"detected_weapon_camera_{camera_no}.jpg"
                            cv2.imwrite(image_path, img)

                            # Trigger alerts
                            send_email_alert()
                            play_sound_alert()
                            send_to_directus(image_path, camera_no)

                            # Remove the saved image after sending
                            os.remove(image_path)

                            weapon_detected = False

        cv2.imshow(f"Camera {camera_no}", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_detection(0, 1)  
