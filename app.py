import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("Real-Time Emotion Detection")

device = torch.device("cpu")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features

    # MUST MATCH TRAINING ARCHITECTURE
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 7)
    )

    model.load_state_dict(
        torch.load("emotion_resnet18_finetuned.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    return model


model = load_model()

# -------------------------
# CLASS LABELS
# -------------------------
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------
# FACE DETECTOR
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# VIDEO PROCESSOR
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            try:
                face_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)

                emotion = class_names[predicted.item()]
                intensity = round(confidence.item() * 100, 2)

                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                label = f"{emotion} ({intensity}%)"
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            except Exception:
                pass

        return img


# -------------------------
# STABLE ICE CONFIGURATION
# -------------------------
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------------
# START WEBRTC
# -------------------------
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
