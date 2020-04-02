import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

#Create a Flask app instance
app = Flask(_name_)

from keras.models import model_from_json, load_model
from keras.backend import set_session
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#Load model
print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
# load json and create model
json_file = open('model_pkm.json', 'r')
json_string = json_file.read()
json_file.close()
model = model_from_json(json_string)
# load weights into new model
model.load_weights('model_pkm.h5')
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded")
global graph
graph = tf.get_default_graph()

#Interaction
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')
	
#Prediction
@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    img = cv2.imread(filename, 1)
    #Step 2
    img = cv2.resize(img, (96,96))
    img = img / 255.0
    #Step 3
    with graph.as_default():
      set_session(sess)
      probabilities = model.predict(img.reshape((1, 96,96,3)))[0,:]
      print(probabilities)
      #Step 4
      poke_label_dict = {'Mewtwo': 0,
                         'Pikachu': 1,
                         'Charmander': 2,
                         'Bulbasaur': 3,
                         'Squirtle': 4,
                         'Psyduck': 5,
                         'Spearow': 6,
                         'Fearow': 7,
                         'Dratini': 8,
                         'Aerodactyl': 9,
                         'Rapidash': 10,
                         'Shellder': 11,
                         'Ninetales': 12,
                         'Pidgey': 13,
                         'Mankey': 14,
                         'Machamp': 15,
                         'Sandslash': 16,
                         'Raichu': 17,
                         'Muk': 18,
                         'Lapras': 19,
                         'Primeape': 20,
                         'Marowak': 21,
                         'Exeggutor': 22,
                         'Meowth': 23,
                         'Raticate': 24,
                         'Snorlax': 25,
                         'Rhyhorn': 26,
                         'Growlithe': 27,
                         'Kingler': 28,
                         'Vaporeon': 29,
                         'Nidoking': 30,
                         'Gyarados': 31,
                         'Vulpix': 32,
                         'Scyther': 33,
                         'Mew': 34,
                         'Seadra': 35,
                         'Vileplume': 36,
                         'Kakuna': 37,
                         'Electrode': 38,
                         'Golbat': 39,
                         'Dewgong': 40,
                         'Lickitung': 41,
                         'Wigglytuff': 42,
                         'Tauros': 43,
                         'Rattata': 44,
                         'Sandshrew': 45,
                         'Nidoqueen': 46,
                         'Kabutops': 47,
                         'Butterfree': 48,
                         'Venusaur': 49}
      poke_list = list(poke_label_dict.keys())
      index = np.argsort(probabilities)
      predictions = {
                "class1":poke_list[index[49]],
                "class2":poke_list[index[48]],
                "class3":poke_list[index[47]],
                "prob1":probabilities[index[49]],
                "prob2":probabilities[index[48]],
                "prob3":probabilities[index[47]],
      }
      #step 5
      return render_template('predict.html', predictions=predictions)           

#Run app 	
app.run(host='0.0.0.0', port=80)
