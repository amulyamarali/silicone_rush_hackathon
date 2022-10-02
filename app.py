# from flask import Flask, render_template, request,redirect, session, url_for

# app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template('Home.html')

# @app.route('/')
# def contact():
#     return render_template('Contact_Us.html')


# if __name__ == "__main__":
#     app.run()

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
# from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'FIRE', 1 : 'NO FIRE'}

cnn = load_model('cnn.h5')

cnn.make_predict_function()

# def predict_label(img_path):
# 	i = tf.keras.utils.load_img(img_path)
# 	i = tf.keras.preprocessing.image.img_to_array(i)
# 	# i = i.reshape(0, 64, 64, 3)
# 	i = np.expand_dims(i, axis = 0)
# 	p = cnn.predict(i)
# 	return dic[p[0]]

def predict_label(img_path):
	i = tf.keras.utils.load_img(img_path, target_size=(64,64))
	i = tf.keras.preprocessing.image.img_to_array(i)
	# i = i.reshape(1,100,100,3)
	i = np.expand_dims(i, axis = 0)
	i = i.reshape(1, 64,64,3)
	# i = np.array(i.getdata()).reshape(i.size[0], i.size[1], 3)
	# i = i.reshape(1, 1,64, 64)
	p = cnn.predict(i)
	return dic[p[0][0]]

# routes

@app.route("/predict", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/")
def m():
	return render_template("home_final.html")

# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/threat")
def threat():
	return render_template("Threat_analyzer.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(port="7000")
