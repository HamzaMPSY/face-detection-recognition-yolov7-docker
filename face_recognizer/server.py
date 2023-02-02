import os
from io import BytesIO

import face_recognition
import flask
import numpy as np
import werkzeug
from PIL import Image

dataset = 'dataset' 					# Dataset Folder
app = flask.Flask(__name__)
faces = []
names = []


def who_is_it(embeeds):
    global faces, names
    matches = face_recognition.compare_faces(faces, embeeds)
    name = "Unknown"
    face_distances = face_recognition.face_distance(faces, embeeds)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = names[best_match_index]

    return name


def load_dataset():
    global faces, names
    for person in os.listdir('../dataset'):
        print(person)
        for pic in os.listdir('../dataset' + '/' + person):
            img = face_recognition.load_image_file(
                '../dataset' + '/' + person + '/' + pic)
            faces.append(face_recognition.face_encodings(img)[0])
            names.append(person)


# @app.route('/add', methods=['POST'])
# def add():
#     # Extract Parametres from the request
#     imagefile = flask.request.files['img']
#     name = flask.request.form['name']
#     name = name.replace(' ', '_')
#     filename = werkzeug.utils.secure_filename(imagefile.filename)
#     print("\n[!]Received image File name : " + imagefile.filename)
#     # # If the the name has no directoryto store the image in we create new one
#     # if not os.path.isdir(dataset + '/' + name):
#     #     os.mkdir(dataset + '/' + name)
#     # # Save the image in the right folder
#     # imagefile.save(dataset + '/' + name + '/' + filename)
#     # # Add it to the database of encodings so we can compare new images with the new person added
#     # if name not in database:
#     #     database[name] = []
#     # img = cv2.imread(dataset + '/' + name + '/'+filename)
#     # img = face_from_image(img)
#     # if img is not None:
#     #     database[name].append(l2_normalize(img_to_encoding(img)[0]))
#     #     return "Image add to " + name
#     return "Image add to " + name + ',But with no face in it!'


@app.route('/predict', methods=['POST'])
def predict():
    imagefile = flask.request.files['img']
    image = Image.open(BytesIO(imagefile.read())).convert('RGB')
    image = np.array(image)
    encoding = face_recognition.face_encodings(image)[0]
    return who_is_it(encoding)


@app.route('/', methods=['GET'])
def home():
    return "Hi its working!"


@app.before_first_request
def initialize():
    print("[*] Please wait until all model are loaded")
    load_dataset()
    print('[+] Database is constructed, you can now send requests')


app.run(host="0.0.0.0", debug=True)
