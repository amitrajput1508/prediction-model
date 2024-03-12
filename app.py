# Add the necessary imports at the beginning
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
with open('your_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('templates/index.html')  # Updated path

# Modify the '/predict' route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = float(request.form['input'])

        # Make a prediction based on user input
        user_prediction = model.predict([[user_input]])

        # Render the prediction on the result page
        return render_template('templates/result.html', prediction=user_prediction[0][0])  # Updated path

    except Exception as e:
        return render_template('templates/result.html', error=str(e))  # Updated path

if __name__ == '__main__':
    app.run(debug=True)
