import gradio as gr
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("model.pkl")

def predict_price(area, bedrooms, bathrooms):
    # Prepare the input data in a DataFrame
    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms]
    })
    # Perform prediction using the loaded model
    predicted_price = model.predict(input_data)[0]
    # Format and return the predicted price
    return f"Predicted Price: ${predicted_price:.2f}"

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.inputs.Number(label="Area (sq ft)"),
        gr.inputs.Number(label="Bedrooms"),
        gr.inputs.Number(label="Bathrooms")
    ],
    outputs="text",
    title="Housing Price Prediction App",
    description="Enter details of the house to predict the housing price."
)

if __name__ == "__main__":
    iface.launch()
