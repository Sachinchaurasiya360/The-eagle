# Project Overview

This project is developed by Team Code Hunter, and it is designed for rajashtan police for hackathon which is conducted by them.You can track any person by using this code and also if there is some suspicious activity or any type of weapon is detected and then it will send an email and trigger the buzzer(All the feature is tested by the jury on real gun in RPH hackathon).The video of the recording also will be store.

## Team Members

- Sachin Chaurasiya (Team Lead, Backend Engineer)


## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Sachinchaurasiya360/code-hunter
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

1. Collect the Faces Dataset by running:
   ```bash
   python get_faces_from_camera_tkinter.py
   ```

2. Convert the dataset into features by running:
   ```bash
   python features_extraction_to_csv.py
   ```

3. To take track, run:
   ```bash
   python attendance_taker.py
   ```

4. Check the Database by running:
   ```bash
   python app.py
   ```
5. Run the flask app for checking website:
   ```bash
   python flaskapp.py
   ```

Our trained model which is best.pt it is pt file so make sure it is valid in your system beacuse sometime it crash during the downloading.



khdx gokg ofdz oblz
