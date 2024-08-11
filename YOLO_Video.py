from ultralytics import YOLO
import cv2
import math
import torch;
def video_detection(path_x):
    video_capture = path_x
    cap=cv2.VideoCapture(video_capture)
            #reseting the video capture to the start
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    
            #use for saving the video
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    

    model=YOLO("../YOLO-Weights/best.pt")   #loading the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda' )
    model.to(device)
    
    
    classNames = ['Rifel', 'Hand Gun', 'Knife', 'weapon','test','detecting','gun']
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                #print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color=(0,255,0)
                if conf>0.70:
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
       
