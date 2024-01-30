from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
import sqlite3
from datetime import datetime

from wtforms import (
    FileField,
    SubmitField,
    StringField,
    DecimalRangeField,
    IntegerRangeField,
)
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os
import cv2
import asyncio

from YOLO_Video import video_detection


app = Flask(__name__)

app.config["SECRET_KEY"] = "rajpolice"
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

@app.route("/track", methods=["GET", "POST"])
def track():
    session.clear()
    return render_template("track.html")

@app.route("/trackperson", methods=["GET", "POST"])
def trackpperson():
    session.clear()
    return render_template("trackperson.html")

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)


@app.route("/FrontPage", methods=["GET", "POST"])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                app.config["UPLOAD_FOLDER"],
                secure_filename(file.filename),
            )
        )
        session["video_path"] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config["UPLOAD_FOLDER"],
            secure_filename(file.filename),
        )
    return render_template("videoprojectnew.html", form=form)


@app.route("/video")
def video():
    return Response(
        generate_frames(path_x=session.get("video_path", None)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/webapp")
def webapp():
    return Response(
        generate_frames_web(path_x=0),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    
@app.route("/webap") #external camera
def webap():
    return Response(
        generate_frames_web(path_x=1),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
@app.route("/cam2")
def cam2():
    return Response(
        generate_frames_web(path_x=2),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    #for last camera
@app.route("/cam3")
def cam3():
     return Response(
         generate_frames_web(path_x=3),
         mimetype="multipart/x-mixed-replace; boundary=frame",
     )


if __name__ == "__main__":
    app.run(debug=True)