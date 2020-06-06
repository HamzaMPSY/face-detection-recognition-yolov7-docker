import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.backend import set_session
import os
import cv2
import flask
import werkzeug
import warnings
warnings.filterwarnings("ignore")

# Global variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataset = 'dataset' 					# Dataset Folder
test = 'test'							# Folder to store all picture of testing the model
app = flask.Flask(__name__)
model = None
margin = 10
database = {}
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # model to extract face from picture
# TF sessiona and graph to use the same model with out load it every time you use it 
sess = tf.Session()
graph = tf.get_default_graph()

def l2_normalize(x, axis=-1, epsilon=1e-10):
	"""
	function that normalize an np.array 
	"""
	output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
	return output

def preprocess(x):
	"""
	function to preprocess an image or array of images
	"""
	if x.ndim == 4:
	    axis = (1, 2, 3)
	    size = x[0].size
	elif x.ndim == 3:
	    axis = (0, 1, 2)
	    size = x.size
	else:
	    raise ValueError('Dimension should be 3 or 4')

	mean = np.mean(x, axis=axis, keepdims=True)
	std = np.std(x, axis=axis, keepdims=True)
	std_adj = np.maximum(std, 1.0/np.sqrt(size))
	y = (x - mean) / std_adj
	return y

def img_to_encoding(img):
	"""
	Encode an image with the facenet model after preprocessing it
	"""
	global model,graph,sess
	x_train = np.array([img])
	x_train = preprocess(x_train)
	with graph.as_default():
		set_session(sess)
		embedding = model.predict_on_batch(x_train)
	return embedding

def loadModel():
	"""
	Load the model
	"""
	global graph,sess
	with graph.as_default():
		set_session(sess)
		loaded_model = load_model('facenet_keras.h5')
		loaded_model.load_weights('facenet_keras_weights.h5')
	return loaded_model

def face_from_image(img):
	"""
	extract the face from the image
	"""
	global face_cascade,margin
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces) > 0 :
		x,y,w,h = faces[0]
		img = img[y-margin//2:y+h+margin//2,
		x-margin//2:x+w+margin//2, :]
		img = cv2.resize(img,(160,160))
		return img
	return None


def who_is_it(image):
	global database,graph,sess
	ths = 0.7 # to_tune
	img = cv2.imread(test + '/'+ image)
	img = face_from_image(img)
	if img is not None:
		with graph.as_default():
			set_session(sess)
			encoding = l2_normalize(img_to_encoding(img))
		min_dist = 100
		identity = "Unknown"
		for (name, db_enc) in database.items():
			for enc in db_enc:
				dist = np.linalg.norm(encoding - enc)
				if dist < min_dist:
					min_dist = dist
					identity = name
		if min_dist > ths:
			identity = "Unknown"
		return identity
	return "This pic does not contain any face!"

@app.route('/add', methods = ['POST'])
def add():
	# Extract Parametres from the request 
	imagefile = flask.request.files['img']
	name = flask.request.form['name']
	name = name.replace(' ','_')
	filename = werkzeug.utils.secure_filename(imagefile.filename)
	print("\n[!]Received image File name : " + imagefile.filename)
	# If the the name has no directoryto store the image in we create new one
	if not os.path.isdir(dataset +'/' + name):
		os.mkdir(dataset +'/' + name)
	# Save the image in the right folder
	imagefile.save(dataset +'/' + name + '/' +filename)
	# Add it to the database of encodings so we can compare new images with the new person added
	if name not in database:
		database[name] = []
	img = cv2.imread(dataset + '/' + name + '/'+filename)
	img = face_from_image(img)
	if img is not None:
		database[name].append(l2_normalize(img_to_encoding(img)[0]))
		return "Image add to "+name
	return "Image add to "+name +',But with no face in it!'

@app.route('/predict', methods = ['POST'])
def predict():
	imagefile = flask.request.files['img']
	filename = werkzeug.utils.secure_filename(imagefile.filename)
	print("\n[!]Received image File name : " + imagefile.filename)
	imagefile.save(test +'/' +filename)
	identity = who_is_it(filename)
	return "it's " + str(identity) + "!"

@app.before_first_request
def initialize():
	global model,database    
	# Here the server begin
	print("[*] Please wait until all model are loaded")
	model = loadModel()
	print("[+] Facenet model is loaded, Database construction is begin, Please wait")
	for sub in os.listdir(dataset):
		database[sub] = []
		for pic_name in os.listdir('dataset/'+sub):
		    img = cv2.imread('dataset/'+sub + '/'+pic_name)
		    img = face_from_image(img)
		    if img is not None:
		        database[sub].append(l2_normalize(img_to_encoding(img)[0]))
	print('[+] Database is constructed, you can now send requests')
app.run(host="0.0.0.0", port=8000, debug=True)