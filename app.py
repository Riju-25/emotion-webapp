import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# -------------------------
# Device
# -------------------------
device = torch.device("cpu")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model = torchvision.models.resnet18(weights=None)
    num_features = model.fc.in_features
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

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# Video Processor
# -------------------------
class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            input_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            emotion = class_names[pred.item()]
            intensity = confidence.item() * 100

            # Draw bounding box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            # Label below box
            label = f"{emotion} {intensity:.1f}%"
            cv2.putText(img,
                        label,
                        (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

        return img

# -------------------------
# Streamlit UI
# -------------------------
st.title("Real-Time Emotion Detection")

rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=rtc_configuration
)
