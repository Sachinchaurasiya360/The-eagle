from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from datetime import datetime

from wtforms import (
    FileField,
    SubmitField
)
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os
import cv2

from YOLO_Video import video_detection


app = Flask(__name__)

app.config["SECRET_KEY"] = "secret"
app.config["UPLOAD_FOLDER"] = "static/files"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x=""):
    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)

        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def generate_frames_web(path_x):
    yolo_output = video_detection(path_x,)

    for detection_ in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)

        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
def login():
    session.clear()
    return render_template("login.html")


@app.route("/home", methods=["GET", "POST"])
def home():
    session.clear()
    return render_template("indexproject.html")


@app.route("/webcam", methods=["GET", "POST"])
def webcam():
    session.clear()
    return render_template("ui.html")



@app.route("/internal_camera")
def internal_camera():
    return Response(
        generate_frames_web(path_x=0),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    
@app.route("/external_camera") #external camera
def external_camera():
    return Response(
        generate_frames_web(path_x=1),
        mimetype="multipart/x-mixed-replace; boundary=frame",

     )

if __name__ == "__main__":
    app.run(debug=True,port=5001)