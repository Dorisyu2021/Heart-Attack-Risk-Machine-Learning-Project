# Machine Learning Project - Team Squirtle

## Project Proposal
* We will use heart attack risk factor data
* We will create a machine learning model that can predict (to some extent) heart attack risk based on risk factors. We will use several different machine learning techniques and pick the most accurate technique to use for our model.
* In this process, we will also attempt to determine the most significant risk factors - does cholesterol level matter more than age, for example.
* Finally, we will create a web-based tool based on our model.

## Project Deployment on Heroku
Website where our application can be accessed:
* https://bootcamp-project-4.herokuapp.com

## Data Sources
Information on the data that we used:
* https://www.kaggle.com/fedesoriano/heart-failure-prediction/version/1

## Repository Organization
* app.py (flask application)
* Procfile (file needed for Heroku)
* requirement.txt (file needed for Heroku)
* Heart Disease Prediction_Machine Learning Integration.pdf (presentation slides)
* **static**
	* best_model.h5 (neural network model file)
	* best_nn_scaler.pkl (data scaling file)
	* style.css (webpage style sheet)
* **templates**
	* index.html (landing page)
	* form_page.html (form input page)
	* prediction_page.html (model output page)
* **model_investigation**
	* heart_attack_risk_knn_model.ipynb (k-nearest neighbors investigation)
	* heart_attack_risk_logistic.ipynb (logistical regression investigation)
	* heart_attack_risk_rfc_model.ipynb (random forest investigation)
	* NN.ipynb (tensorflow investigation)
	* NN_keras.ipynb (tensorflow with keras investigation)
	* **data**
		* heart.csv (raw data file used for training and testing)
