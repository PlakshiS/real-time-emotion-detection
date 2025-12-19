import gradio as gr
import cv2
import numpy as np
from keras.models import model_from_json

# Load the model architecture from json
json_file = open('my_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load the model weights
emotion_model.load_weights('my_model.h5')

# Define emotion labels
EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Preprocess the image using OpenCV
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    preprocessed = resized / 255.0
    preprocessed = np.expand_dims(np.expand_dims(preprocessed, axis=-1), axis=0)
    return preprocessed

# Define prediction function
def predict(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Predict emotion
    preds = emotion_model.predict(preprocessed_image)
    label = EMOTIONS[np.argmax(preds)]
    probs = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, preds[0])}
    return label, probs

# Define Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Your face"),
    outputs=[gr.Label(label="Predicted Emotion"), gr.Label(label="Probabilities")],
    capture_session=True,
    title="Emotion Detection",
    description="This model predicts the emotions of the uploaded images.",
)

if __name__ == "__main__":
    print("Starting Gradio app. Visit http://localhost:7860 to access the app.")
    iface.launch(inbrowser=True)