import numpy as np
import pickle 
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data[key]) for key in data.keys()]
    final_features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(final_features)
    output = prediction[0]

    if output == 1:
        return render_template('result.html', result='The person is not likely to have a heart disease!')
    else:
        return render_template('result.html', result='The person is likely to have a heart disease!')

if __name__ == '__main__':
    app.run()
