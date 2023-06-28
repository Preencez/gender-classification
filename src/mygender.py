import joblib
import gradio as gr
import pandas as pd

# Load the saved Linear Regression model
path = r"D:\Projects\gradio app for gender classification\linear_regression_model.joblib"
linear_regression_model = joblib.load(path)

# Get the feature names from the model
feature_names = linear_regression_model.feature_names_in_

# Function to make predictions using the loaded Linear Regression model
# Function to make predictions using the loaded Linear Regression model
# Function to make predictions using the loaded Linear Regression model
# Function to make predictions using the loaded Linear Regression model
def predict(*features):
    # Create a dictionary to store the input features
    input_features = {name: value for name, value in zip(feature_names, features)}

    # Create a DataFrame from the input features
    df = pd.DataFrame(input_features, index=[0])

    # Make the prediction using the loaded Linear Regression model
    prediction = linear_regression_model.predict(df)[0]

    # Extract the first element of the prediction array
    prediction = prediction[0]

    # Round the prediction to the nearest integer
    prediction = int(round(prediction))

    # Map the rounded prediction to "Male" or "Female"
    if prediction == 0:
        result = "Male"
    else:
        result = "Female"

    # Return the prediction
    return result



# Define the input and output interfaces
input_interface = gr.Interface(
    predict,
    inputs=[
        gr.inputs.Checkbox(label="Long Hair"),
        gr.inputs.Slider(minimum=0, maximum=100, label="Forehead Width (cm)"),
        gr.inputs.Slider(minimum=0, maximum=100, label="Forehead Height (cm)"),
        gr.inputs.Checkbox(label="Wide Nose"),
        gr.inputs.Checkbox(label="Long Nose"),
        gr.inputs.Checkbox(label="Thin Lips"),
        gr.inputs.Checkbox(label="Long Distance from Nose to Lip"),
        gr.inputs.Slider(minimum=0, maximum=1, label="Forehead Height Ratio"),
        gr.inputs.Slider(minimum=0, maximum=1, label="Nose Width Ratio"),
        gr.inputs.Slider(minimum=0, maximum=1, label="Forehead Nose Index"),
        gr.inputs.Slider(minimum=0, maximum=1, label="Forehead Height Nose Index")
    ],
    outputs=gr.outputs.Textbox(label="Prediction")
)

# Launch the interface
input_interface.launch()
