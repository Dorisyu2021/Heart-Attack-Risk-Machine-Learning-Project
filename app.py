# This Flask app utilizes a Neural Network machine learning model to 
# predict the likelihood that a patient is suffering from heart disease.
# The model has already been trained and tested and saved to file: "best_model.h5".
# When the "predict" button is pressed, the model will load and use the input data
# from the user to make a prediction.



# External dependencies:
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
import pickle


# Create an instance of Flask:
app=Flask(__name__)


# Define initial descriptive text:
start_text = "Please complete the above information and then click the predict button."


# Index route (landing page):
@app.route("/")
def home():

    #Status message to terminal and display landing page:
    print("Index page activated.")
    return render_template("index.html")



# Patient input route (prediction page):
@app.route("/form")
def index():

   # Status message to terminal and display prediction page:
    print("Form page activated.")    
    return render_template("form_page.html", message=start_text)


# Heart disease prediction route (makes the prediction):
@app.route("/pred", methods=["POST"])
def predict():

    # Status message to terminal:
    print("Model prediction initiated.")

    # Load the model and scaler from their external folder/files:
    model_folder = "static/best_model.h5"
    scaler_file = "static/best_nn_scaler.pkl"
    loaded_model = tf.keras.models.load_model(model_folder)
    loaded_scaler = pickle.load(open(scaler_file, "rb"))
    print(f"Model loaded from folder: {model_folder}")
    print(f"Scaler loaded from file: {scaler_file}")

    # Create column headers to match the ones used in the model's training dataset:
    column_headers = ["Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak",
                    "Sex_M",
                    "ChestPainType_ASY","ChestPainType_ATA","ChestPainType_NAP","ChestPainType_TA",
                    "RestingECG_LVH","RestingECG_Normal","RestingECG_ST",
                    "ExerciseAngina_Y",
                    "ST_Slope_Down","ST_Slope_Flat","ST_Slope_Up"]

    """
    # Testing data to use if prediction_page.html is unavailable:
    Age = 65
    Sex = "M"
    ChestPainType = "ATA"
    RestingBP = 125
    Cholesterol = 100
    FastingBS = 0
    RestingECG = "Normal"
    MaxHR = 150
    ExerciseAngina = "N"
    Oldpeak = 1.5
    ST_Slope = "Flat"                
    """
    # Get data from posted form:
    if request.method != "POST":
        return render_template("prediction_page.html", message="Error: Please try again.")
    else:
        Age = request.form.get("Age")
        Sex = request.form.get("Gender")
        ChestPainType = request.form.get("ChestPainType")
        RestingBP = request.form.get("RestingBP")
        Cholesterol = request.form.get("Cholesterol")
        FastingBS = request.form.get("FastingBS")
        RestingECG = request.form.get("RestingECG")
        MaxHR = request.form.get("MaxHR")
        ExerciseAngina = request.form.get("ExerciseAngina")
        Oldpeak = request.form.get("Oldpeak")
        ST_Slope = request.form.get("STslope") 

    # Put the input data into a row:
    data_row = []
    data_row.append(Age)
    data_row.append(RestingBP)
    data_row.append(Cholesterol)
    data_row.append(FastingBS)
    data_row.append(MaxHR)
    data_row.append(Oldpeak)
    if Sex == "M":
        data_row.append(1)
    else:
        data_row.append(0)
    if ChestPainType == "ASY":
        data_row.append(1)
        data_row.append(0)
        data_row.append(0)
        data_row.append(0)
    elif ChestPainType == "ATA":
        data_row.append(0)
        data_row.append(1)
        data_row.append(0)
        data_row.append(0)
    elif ChestPainType == "NAP":
        data_row.append(0)
        data_row.append(0)
        data_row.append(1)
        data_row.append(0)    
    else:
        data_row.append(0)
        data_row.append(0)
        data_row.append(0)
        data_row.append(1)
    if RestingECG == "LVH":
        data_row.append(1)
        data_row.append(0)
        data_row.append(0)
    elif RestingECG == "Normal":
        data_row.append(0)
        data_row.append(1)
        data_row.append(0)
    else:
        data_row.append(0)
        data_row.append(0)
        data_row.append(1)
    if ExerciseAngina == "Y":
        data_row.append(1)
    else:
        data_row.append(0)
    if ST_Slope == "Down":
        data_row.append(1)
        data_row.append(0)
        data_row.append(0)
    elif ST_Slope == "Flat":
        data_row.append(0)
        data_row.append(1)
        data_row.append(0)
    else:
        data_row.append(0)
        data_row.append(0)
        data_row.append(1)
        
    # Create a single row dataframe to pass as input to the model:
    input_data = pd.DataFrame([data_row], columns=column_headers)

    # Scale the input data: 
    input_data_scaled = loaded_scaler.transform(input_data)

    # Get prediction from model:
    y = loaded_model.predict(input_data_scaled)

    # Return the prediction result to the user:
    result = f"The likelihood of the patient having heart disease is {100 * y[0][0]:.1f}%"
    print(result)
    message = "For a patient with the following parameters:"
    return render_template("prediction_page.html", 
                            message=message, prediction=result, 
                            age=f"Age:  {Age}", gender=f"Gender:  {Sex}",
                            paintype=f"Chest Pain Type:  {ChestPainType}",
                            restBP=f"Resting Blood Pressure:  {RestingBP}",
                            cholesterol=f"Cholesterol:  {Cholesterol}",
                            bloodsugar=f"Fasting Blood Sugar:  {FastingBS}",
                            ECG=f"Resting Electrocardiogram:  {RestingECG}",
                            HRmax=f"Maximum Heart Rate:  {MaxHR}",
                            exerciseangina=f"Angina INduced by Exercise:  {ExerciseAngina}",
                            STdepression=f"ST Wave Depression:  {Oldpeak}",
                            STslope=f"ST Wave Slope:  {ST_Slope}")
    
    
 
# Run the Flask app:
if __name__=="__main__":
    app.debug=True
    app.run()