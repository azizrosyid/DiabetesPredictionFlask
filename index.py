from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    test_json = request.get_json()
    dataJson = {'Pregnancies': [test_json['pregnancies']], 'Glucose': [test_json['glucose']], 'BloodPressure': [test_json['blood_pressure']], 'SkinThickness': [
        test_json['skin_thickness']],     'Insulin': [test_json['insulin']],   'BMI': [test_json['bmi']],   'DiabetesPedigreeFunction': [test_json['diabetes_pedigree_function']], 'Age': [test_json['age']]}

    data = pd.DataFrame(dataJson)
    lin_reg = joblib.load("./linear_reg.pkl")
    prediction = lin_reg.predict(data)
    classes = ['No Diabetes', 'Diabetes']
    text = classes[prediction[0]]
    return jsonify({"result": text})


if __name__ == '__main__':
    app.run(debug=False)
