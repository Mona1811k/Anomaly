from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file, session
import mysql.connector, os
#Import libraries
import keras
import cv2
import numpy as np 
from keras.models import load_model
# from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders



# Define constants
IMG_SIZE = 224  # Image size for InceptionV3
MAX_SEQ_LENGTH = 20  # Adjusted to match model's expected sequence length
NUM_FEATURES = 2048  # Features from InceptionV3 (features per frame)


# Load the model
model = load_model('GRU.h5')

# Function to load and preprocess a single video
def load_single_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # Convert from BGR to RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Crop the center square of the frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Preprocess the frames (extract features using InceptionV3)
def preprocess_frames(frames):
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    frames = preprocess_input(frames)  # Preprocess the frames
    features = feature_extractor.predict(frames)
    return features



app = Flask(__name__)
app.secret_key = 'sghcdvsadvbdcsbvnv'
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='video_dataset'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/About')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['useremail']
        password = request.form['password']
        c_password = request.form['c_password']
        username = request.form['username']
        age = request.form['age']
        gender = request.form['gender']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password,username,age,gender) VALUES (%s, %s, %s, %s, %s)"
                values = (email, password,username,age, gender)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['useremail']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email
                session["user_email"] = user_email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template("upload.html", message="No file part")

        myfile = request.files['video']

        if myfile.filename == '':
            return render_template("upload.html", message="No selected file")

        # Acceptable video formats
        accepted_formats = ['avi', 'mp4', 'mov', 'mkv']
        if not myfile.filename.split('.')[-1].lower() in accepted_formats:
            message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
            return render_template("upload.html", message=message)

        # Save the uploaded video file
        filename = secure_filename(myfile.filename)
        video_path = os.path.join('static/Video/', filename)
        myfile.save(video_path)

        # Ensure the frames directory exists
        frames_dir = "static/Frames/"
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        # Process the video and save a single random frame
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return render_template("upload.html", message="Error: Video is empty or cannot be read")

        random_frame_index = np.random.randint(0, total_frames)

        saved_frame_path = os.path.join(frames_dir, f"{os.path.splitext(filename)[0]}_frame.jpg")
        current_frame = 0
        success = False  # Track if frame was successfully saved

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame == random_frame_index:
                success = cv2.imwrite(saved_frame_path, frame)
                break

            current_frame += 1

        cap.release()

        if not success:
            print("❌ ERROR: Failed to save frame")
            return render_template("upload.html", message="Error: Failed to save frame")

        print(f" Frame successfully saved at: {saved_frame_path}")

        # Process the video and predict the class
        frames = load_single_video(video_path)
        features = preprocess_frames(frames)

        # Limit the number of frames to 20 (if necessary)
        features = features[:MAX_SEQ_LENGTH]  # Shape: (20, 2048)
        features = features[None, :, :]  # Shape: (1, 20, 2048) (batch_size, sequence_length, num_features)
        mask = np.ones((1, len(features[0])))  # Mask for all frames (1 = not masked)

        # Predict the label using the model
        prediction = model.predict([features, mask])

        # Class labels for video prediction (Normal, Violence, Weaponized)
        class_labels = ['Normal', 'Violence', 'Weaponized']
        predicted_class = class_labels[np.argmax(prediction)]
        print(f"Predicted class: {predicted_class}")

        # Ensure file exists before sending email
        if os.path.exists(saved_frame_path):
            print(f" Frame found: {saved_frame_path}")
            admin_emails = ["msowmya3108@gmail.com", "sowmya.mallela04@gmail.com"]
            for email in admin_emails:
                success = send_email_with_attachment(email, saved_frame_path)
                if success:
                    print(f" Email sent to {email}")
                else:
                    print(f"❌ Email failed for {email}")
        else:
            print(f"❌ ERROR: Image file not found at {saved_frame_path}")

        return render_template('upload.html', prediction=predicted_class, path=video_path)

    return render_template('upload.html')



# Function to send an email alert
import smtplib
from email.message import EmailMessage
import imghdr
import os
import cv2
sender_address = "sowmya.mallela04@gmail.com"
sender_pass = "dlcmdknqtwfwynvf"  # App password dlcmdknqtwfwynvf
def send_email_with_attachment(owner_email, image_path):
    try:
        if not os.path.exists(image_path):
            print(f"❌ ERROR: Image file not found at {image_path}")
            return False

        new_message = EmailMessage()
        new_message['Subject'] = "Smart Surveillance Alert"
        new_message['From'] = sender_address
        new_message['To'] = owner_email
        new_message.set_content('Anomaly action detected. Please check the attached image.')

        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = os.path.basename(f.name)

        if image_type is None:
            print(f"❌ ERROR: Could not determine image type for {image_path}")
            return False

        new_message.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

        print(f" Sending email to {owner_email} with image: {image_name} ({image_type})")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_address, sender_pass)
            smtp.send_message(new_message)

        print(" Email sent successfully!")
        return True

    except smtplib.SMTPAuthenticationError:
        print(" ERROR: SMTP Authentication failed! Check sender email and app password.")
    except Exception as e:
        print(f" ERROR sending email: {e}")

    return False


    
if __name__ == '__main__':
    app.run(debug = True)






# from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
# import mysql.connector, os
# #Import libraries
# import keras
# import cv2
# import numpy as np
# from keras.models import load_model
# from werkzeug.utils import secure_filename


# # Define constants
# IMG_SIZE = 224  # Image size for InceptionV3
# MAX_SEQ_LENGTH = 20  # Adjusted to match model's expected sequence length
# NUM_FEATURES = 2048  # Features from InceptionV3 (features per frame)


# # Load the model
# model = load_model('GRU.h5')

# # Function to load and preprocess a single video
# def load_single_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = crop_center_square(frame)
#             frame = cv2.resize(frame, resize)
#             frame = frame[:, :, [2, 1, 0]]  # Convert from BGR to RGB
#             frames.append(frame)

#             if len(frames) == max_frames:
#                 break
#     finally:
#         cap.release()
#     return np.array(frames)

# # Crop the center square of the frame
# def crop_center_square(frame):
#     y, x = frame.shape[0:2]
#     min_dim = min(y, x)
#     start_x = (x // 2) - (min_dim // 2)
#     start_y = (y // 2) - (min_dim // 2)
#     return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# # Preprocess the frames (extract features using InceptionV3)
# def preprocess_frames(frames):
#     feature_extractor = keras.applications.InceptionV3(
#         weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
#     )
#     preprocess_input = keras.applications.inception_v3.preprocess_input

#     frames = preprocess_input(frames)  # Preprocess the frames
#     features = feature_extractor.predict(frames)
#     return features



# app = Flask(__name__)
# app.secret_key = 'sghcdvsadvbdcsbvnv'

# mydb = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     port="3306",
#     database='video_dataset'
# )

# mycursor = mydb.cursor()

# def executionquery(query,values):
#     mycursor.execute(query,values)
#     mydb.commit()
#     return

# def retrivequery1(query,values):
#     mycursor.execute(query,values)
#     data = mycursor.fetchall()
#     return data

# def retrivequery2(query):
#     mycursor.execute(query)
#     data = mycursor.fetchall()
#     return data


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/About')
# def about():
#     return render_template('about.html')

# @app.route('/register', methods=["GET", "POST"])
# def register():
#     if request.method == "POST":
#         email = request.form['useremail']
#         password = request.form['password']
#         c_password = request.form['c_password']
#         username = request.form['username']
#         age = request.form['age']
#         gender = request.form['gender']
#         if password == c_password:
#             query = "SELECT UPPER(email) FROM users"
#             email_data = retrivequery2(query)
#             email_data_list = []
#             for i in email_data:
#                 email_data_list.append(i[0])
#             if email.upper() not in email_data_list:
#                 query = "INSERT INTO users (email, password,username,age,gender) VALUES (%s, %s, %s, %s, %s)"
#                 values = (email, password,username,age, gender)
#                 executionquery(query, values)
#                 return render_template('login.html', message="Successfully Registered! Please go to login section")
#             return render_template('register.html', message="This email ID is already exists!")
#         return render_template('register.html', message="Conform password is not match!")
#     return render_template('register.html')


# @app.route('/login', methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form['useremail']
#         password = request.form['password']
        
#         query = "SELECT UPPER(email) FROM users"
#         email_data = retrivequery2(query)
#         email_data_list = []
#         for i in email_data:
#             email_data_list.append(i[0])

#         if email.upper() in email_data_list:
#             query = "SELECT UPPER(password) FROM users WHERE email = %s"
#             values = (email,)
#             password__data = retrivequery1(query, values)
#             if password.upper() == password__data[0][0]:
#                 global user_email
#                 user_email = email

#                 return redirect("/home")
#             return render_template('home.html', message= "Invalid Password!!")
#         return render_template('login.html', message= "This email ID does not exist!")
#     return render_template('login.html')


# @app.route('/home')
# def home():
#     return render_template('home.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         if 'video' not in request.files:
#             return render_template("upload.html", message="No file part")

#         myfile = request.files['video']

#         if myfile.filename == '':
#             return render_template("upload.html", message="No selected file")

#         # Acceptable video formats
#         accepted_formats = ['avi', 'mp4', 'mov', 'mkv']
#         if not myfile.filename.split('.')[-1].lower() in accepted_formats:
#             message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
#             return render_template("upload.html", message=message)

#         # Save the uploaded video file
#         filename = secure_filename(myfile.filename)
#         video_path = os.path.join('static/Video/', filename)
#         myfile.save(video_path)

#         # Process the video and predict the class
#         frames = load_single_video(video_path)
#         features = preprocess_frames(frames)

#         # Limit the number of frames to 20 (if necessary)
#         features = features[:MAX_SEQ_LENGTH]  # Shape: (20, 2048)
#         features = features[None, :, :]  # Shape: (1, 20, 2048) (batch_size, sequence_length, num_features)
#         mask = np.ones((1, len(features[0])))  # Mask for all frames (1 = not masked)

#         # Predict the label using the model
#         prediction = model.predict([features, mask])

#         # Class labels for video prediction (Normal, Violence, Weaponized)
#         class_labels = ['Normal', 'Violence', 'Weaponized']
#         predicted_class = class_labels[np.argmax(prediction)]
#         print(f"Predicted class: {predicted_class}")


#         # Return the prediction along with the video path to the template
#         return render_template('upload.html', prediction=predicted_class, path=video_path)

#     return render_template('upload.html')

# if __name__ == '__main__':
#     app.run(debug = True)