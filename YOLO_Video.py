# from ultralytics import YOLO
# import cv2
# import math
# import torch;
# def video_detection(path_x):
#     video_capture = path_x
#     cap=cv2.VideoCapture(video_capture)
#             #reseting the video capture to the start
#     frame_width=int(cap.get(3))
#     frame_height=int(cap.get(4))
    
#             #use for saving the video
#     #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    

#     model=YOLO("../YOLO-Weights/best.pt")   #loading the model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cuda' )
#     model.to(device)
    
    
#     classNames = ['Rifel', 'Hand Gun', 'Knife', 'weapon','test','detecting','gun']
#     while True:
#         success, img = cap.read()
#         results=model(img,stream=True)
#         for r in results:
#             boxes=r.boxes
#             for box in boxes:
#                 x1,y1,x2,y2=box.xyxy[0]
#                 x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
#                 #print(x1,y1,x2,y2)
#                 conf=math.ceil((box.conf[0]*100))/100
#                 cls=int(box.cls[0])
#                 class_name=classNames[cls]
#                 label=f'{class_name}{conf}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 print(t_size)
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 color=(0,255,0)
#                 if conf>0.70:
#                     cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
#                     cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
#                     cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

#         yield img
       
       
       
#from ultralytics import YOLO
# import cv2
# import math
# import torch
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# def send_email_alert():
#     # Set up the SMTP server details
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 587
#     sender_email = "mrsachinchaurasiya@gmail.com"  # Replace with your email address
#     sender_password = "uqoo ngyq beko puys"  # Replace with your email password
#     receiver_email = "Sachinchaurasiya69@gmail.com"  # Replace with the receiver's email address
    
#     # Create the email content
#     subject = "Weapon Detected Alert"
#     body = "A weapon has been detected in the video stream."
    
#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = subject
#     msg.attach(MIMEText(body, 'plain'))
    
#     # Send the email
#     try:
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(sender_email, sender_password)
#         text = msg.as_string()
#         server.sendmail(sender_email, receiver_email, text)
#         server.quit()
#         print("Email alert sent successffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffully.")
#     except Exception as e:
#         print(f"Failed to send email alert: {e}")

# def video_detection(path_x):
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)

#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     model = YOLO("../YOLO-Weights/best.pt")  # Loading the model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

#                     # Trigger alert if weapon is detected
#                     if class_name.lower() in ['rifle', 'hand gun', 'knife', 'weapon', 'gun']:
#                         weapon_detected = True

#         if weapon_detected:
#             send_email_alert()
#             weapon_detected = False  # Reset to avoid multiple alerts for the same detection

#         yield img

# # Example usage
# # video_detection("path_to_video.mp4")

from ultralytics import YOLO
import cv2
import math
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from playsound import playsound  # Import the playsound library

def send_email_alert():
    # Set up the SMTP server details
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "mrsachinchaurasiya@gmail.com"  # Replace with your email address
    sender_password = "uqoo ngyq beko puys"  # Replace with your email password
    receiver_email = "Sachinchaurasiya69@gmail.com"  # Replace with the receiver's email address
    
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



import pygame

def play_sound_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("aaa.mp3")  # Replace with the path to your alert sound file
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for the sound to finish playing
            continue
        print("Sound alert played successfully.")
    except Exception as e:
        print(f"Failed to play sound alert: {e}")



# def play_sound_alert():
#     # Play the alert sound
#     try:
#         playsound("aaa.mp3")  # Replace with the path to your alert sound file
#         print("Sound alert played successfully.")
#     except Exception as e:
#         print(f"Failed to play sound alert: {e}")

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("../YOLO-Weights/best.pt")  # Loading the model
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

                    # Trigger alert if weapon is detected
                    if class_name.lower() in ['rifle', 'hand gun', 'knife', 'weapon', 'gun']:
                        weapon_detected = True

        if weapon_detected:
            send_email_alert()
            play_sound_alert()  # Play the sound alert
            weapon_detected = False  # Reset to avoid multiple alerts for the same detection

        yield img

# Example usage
# video_detection("path_to_video.mp4")
