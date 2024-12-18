# from ultralytics import YOLO
# import cv2
# import math
# import torch
# import smtplib
# import pygame
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase
# from email import encoders
# import os

# def play_sound_alert():
#     try:
#         pygame.mixer.init()
#         pygame.mixer.music.load("Alert_sound.mp3")
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             continue
#         print("Sound alert played successfully.")
#     except Exception as e:
#         print(f"Failed to play sound alert: {e}")

# def send_email_alert(image_path):
#     try:
#         sender_email = "mrsachinchaurasiya@gmail.com"
#         sender_password = "khdx gokg ofdz oblz"
#         recipient_email = "sachinchaurasiya69@gmail.com"
#         subject = "Weapon Detected Alert"
#         body = "A weapon has been detected in the video feed. Please find the attached image."
        
#         # Create email
#         msg = MIMEMultipart()
#         msg['From'] = sender_email
#         msg['To'] = recipient_email
#         msg['Subject'] = subject
#         msg.attach(MIMEText(body, 'plain'))
        
#         # Attach image
#         with open(image_path, 'rb') as attachment:
#             part = MIMEBase('application', 'octet-stream')
#             part.set_payload(attachment.read())
#         encoders.encode_base64(part)
#         part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
#         msg.attach(part)
        
#         # Send email
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()
#         server.login(sender_email, sender_password)
#         server.send_message(msg)
#         server.quit()
        
#         print("Email alert sent successfully.")
#     except Exception as e:
#         print(f"Failed to send email: {e}")

# def video_detection(path_x):
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)

#     model = YOLO("../YOLO-Weights/Weapon.pt")  # Loading the model
#     device = torch.device('cuda')
#     model.to(device)

#     classNames = ['Rifle', 'Hand Gun', 'Knife', 'Weapon', 'Test', 'Detecting', 'Gun']
#     weapon_detected = False

#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         results = model(img, stream=True)
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 cls = int(box.cls[0])
#                 class_name = classNames[cls]
#                 label = f'{class_name} {conf}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 color = (0, 255, 0)

#                 if conf > 0.70:
#                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#                     cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Filled
#                     cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
#                     if class_name.lower() in ['rifle', 'hand gun', 'knife', 'weapon', 'gun']:
#                         weapon_detected = True
#                         if weapon_detected:
#                             # Save the current frame
#                             image_path = "detected_weapon_frame.jpg"
#                             cv2.imwrite(image_path, img)
                            
#                             # Send email alert with the frame
#                             send_email_alert(image_path)
                            
#                             # Play sound alert
#                             play_sound_alert()
#                             weapon_detected = False

#         yield img

from ultralytics import YOLO
import cv2
import math
import torch
import smtplib
import pygame
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os       
import requests

direct_us_url = "http://172.27.229.241:8055"

def upload_frame_to_directus(image_path):
    """
    Uploads the image frame to Directus and returns the file ID.
    """
    try:
        with open(image_path, 'rb') as f:
            s = requests.Session()
            headers = {"Authorization": "Bearer 7k8V9rO8RtGRTYAeaN-NB2jVQ-DP9fzL"}  # Replace with your Directus API token
            response = s.post(
                f"{direct_us_url}/files",
                headers=headers,
                files={"file": ("image.jpg", f, "image/jpeg")},
                data={"folder": "9ec2c9f6-3e72-432b-8173-5893fb3a63d2"}  # Replace with your folder ID
            ).json()
            if "data" in response:
                file_id = response["data"]["id"]
                print(f"Image uploaded successfully. File ID: {file_id}")
                return file_id
            else:
                print(f"Failed to upload image: {response}")
                return None
    except Exception as e:
        print(f"Error while uploading image to Directus: {e}")
        return None

def send_directus_alert(location, file_id):
    """
    Sends an alert to the Directus instance with location and uploaded frame reference.
    """
    try:
        s = requests.Session()
        headers = {"Authorization": "Bearer 7k8V9rO8RtGRTYAeaN-NB2jVQ-DP9fzL"}  # Replace with your Directus API token
        payload = {
            "location": location,
            "frame": file_id,
        }
        response = s.post(f"{direct_us_url}/items/weapon_alert", headers=headers, json=payload).json()
        if "data" in response:
            print("Alert sent to Directus successfully.")
        else:
            print(f"Failed to send alert: {response}")
    except Exception as e:
        print(f"Error while sending alert to Directus: {e}")



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


def video_detection(path_x):
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
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = (0, 255, 0)

                if conf > 0.70:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    if class_name.lower() in ['rifle', 'hand gun', 'knife', 'weapon', 'gun']:
                        weapon_detected = True
                        if weapon_detected:
                            # Save the current frame
                            image_path = "detected_weapon_frame.jpg"
                            cv2.imwrite(image_path, img)
                            
                            # Upload frame to Directus
                            file_id = upload_frame_to_directus(image_path)
                            
                            # If upload is successful, send alert
                            if file_id:
                                send_directus_alert(location="Bhopal", file_id=file_id)
                            
                            # Send email alert with the frame
                            # send_email_alert(image_path)
                            
                            # Play sound alert
                            play_sound_alert()
                            
                            weapon_detected = False

        yield img

   