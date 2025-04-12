import streamlit as st
import torch
import cv2
import numpy as np
import time
from collections import deque
from torchvision import transforms
import os
import mediapipe as mp
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Drowsiness Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CNN-LSTM model class (this must match your trained model architecture)
class CNN_LSTM(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM, self).__init__()

        # CNN for feature extraction
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        # LSTM to process temporal features
        self.lstm = torch.nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Classification head
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, c, h, w)

        # Extract CNN features
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # Process with LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Take the output of the last time step
        last_time_step = lstm_out[:, -1, :]

        # Classification
        output = self.fc(last_time_step)

        return output

# Define DrowsinessCNN model class
class DrowsinessCNN(torch.nn.Module):
    def __init__(self, input_size=112):
        super(DrowsinessCNN, self).__init__()
        assert input_size == 112
        self.conv_layers = torch.nn.Sequential(
            # First conv block
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Third conv block
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth conv block
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.feature_size = 256 * 7 * 7

        # Fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return torch.sigmoid(x).squeeze(1)

# Define BlurDetector class (based on ZzzSafeRealTime)
class BlurDetector:
    def __init__(self):
        # Thresholds for blur detection methods (you'll need to tune these)
        self.laplacian_threshold = 100  # Lower values indicate more blur
        self.tenengrad_threshold = 20000
        self.sobel_threshold = 15

        # Frame history for smoothing predictions (avoid flickering)
        self.prediction_history = deque(maxlen=10)

        # Stats for analysis
        self.blur_scores = {
            "laplacian": deque(maxlen=100),
            "tenengrad": deque(maxlen=100),
            "sobel": deque(maxlen=100)
        }

    def variance_of_laplacian(self, img):
        """Calculate the Laplacian variance to detect blur."""
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def tenengrad(self, img):
        """Calculate Tenengrad score for blur detection."""
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        return np.sqrt(gx**2 + gy**2).sum()

    def sobel_blur_score(self, img):
        """Calculate Sobel-based blur score."""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(np.sqrt(sobelx**2 + sobely**2))

    def detect_drowsiness(self, frame):
        """Detect drowsiness by analyzing frame blurriness."""
        # Convert to grayscale for blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate blur scores using multiple methods
        lap_score = self.variance_of_laplacian(gray)
        ten_score = self.tenengrad(gray)
        sob_score = self.sobel_blur_score(gray)

        # Store scores for visualization
        self.blur_scores["laplacian"].append(lap_score)
        self.blur_scores["tenengrad"].append(ten_score)
        self.blur_scores["sobel"].append(sob_score)

        # Normalize scores for visualization (0-1 range)
        lap_norm = min(1.0, max(0.0, lap_score / (self.laplacian_threshold * 2)))
        ten_norm = min(1.0, max(0.0, ten_score / (self.tenengrad_threshold * 2)))
        sob_norm = min(1.0, max(0.0, sob_score / (self.sobel_threshold * 2)))
        
        # Drowsy detection logic - multiple methods combined
        is_drowsy = (
            lap_score < self.laplacian_threshold or
            ten_score < self.tenengrad_threshold or
            sob_score < self.sobel_threshold
        )

        # Add to history for smoothing
        self.prediction_history.append(is_drowsy)

        # Smooth prediction (avoid flickering)
        drowsy_ratio = sum(self.prediction_history) / len(self.prediction_history)
        final_prediction = drowsy_ratio > 0.6  # Consider drowsy if >60% of recent frames indicate drowsiness

        # Return prediction and normalized drowsiness probability (1.0 = drowsy, 0.0 = alert)
        # Invert since higher values should mean more drowsy for consistent UI
        drowsy_prob = drowsy_ratio
        
        # Return detailed scores for debugging
        scores = {
            "laplacian": lap_score,
            "tenengrad": ten_score,
            "sobel": sob_score,
            "laplacian_norm": lap_norm,
            "tenengrad_norm": ten_norm,
            "sobel_norm": sob_norm
        }
        
        return 1 if final_prediction else 0, drowsy_prob, scores
    
class MediaPipeDrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        
        # Constants
        self.EAR_THRESHOLD = 0.24
        self.MAR_THRESHOLD = 0.6
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.OUTER_MOUTH = [61, 81, 311, 308, 402, 314, 317, 87]
        
        # State tracking
        self.eye_closed_frames = 0
        self.eye_closed_threshold = 3

    def calculate_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth):
        A = distance.euclidean(mouth[1], mouth[7])
        B = distance.euclidean(mouth[2], mouth[6])
        C = distance.euclidean(mouth[3], mouth[5])
        D = distance.euclidean(mouth[0], mouth[4])
        return (A + B + C) / (3.0 * D)

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        # Initialize default values
        drowsy = 0
        drowsy_prob = 0.0
        status = "No face detected"
        ear = 0.0
        mar = 0.0
        processed_frame = frame.copy()  # Return original frame if no face
        
        if results.multi_face_landmarks:
            try:
                landmarks = results.multi_face_landmarks[0]  # Get first face
                
                # Get coordinates - with proper error handling
                def get_coords(indices):
                    return [(int(landmarks.landmark[i].x * frame.shape[1]), 
                            int(landmarks.landmark[i].y * frame.shape[0])) 
                            for i in indices if i < len(landmarks.landmark)]
                
                left_eye = get_coords(self.LEFT_EYE)
                right_eye = get_coords(self.RIGHT_EYE)
                mouth = get_coords(self.OUTER_MOUTH)
                
                # Only proceed if we got all required landmarks
                if len(left_eye) == 6 and len(right_eye) == 6 and len(mouth) >= 8:
                    # Calculate metrics
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    mar = self.calculate_mar(mouth[:8])  # Ensure we use first 8 mouth points
                    
                    # Determine status
                    if ear < self.EAR_THRESHOLD and mar > self.MAR_THRESHOLD:
                        status = "Drowsy + Yawning"
                        drowsy_prob = 0.9
                    elif ear < self.EAR_THRESHOLD:
                        status = "Drowsy"
                        drowsy_prob = 0.7
                    elif mar > self.MAR_THRESHOLD:
                        status = "Yawning"
                        drowsy_prob = 0.6
                    else:
                        status = "Alert"
                        drowsy_prob = 0.1
                    
                    # Update frame counter for eye closure
                    if ear < self.EAR_THRESHOLD:
                        self.eye_closed_frames += 1
                    else:
                        self.eye_closed_frames = max(0, self.eye_closed_frames - 1)
                        
                    if self.eye_closed_frames >= self.eye_closed_threshold:
                        drowsy_prob = max(drowsy_prob, 0.8)
                    
                    drowsy = 1 if drowsy_prob > 0.5 else 0
                    
                    # Draw landmarks on frame
                    for x, y in left_eye + right_eye + mouth[:8]:  # Only first 8 mouth points
                        cv2.circle(processed_frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Add text overlay
                    cv2.putText(processed_frame, f"EAR: {ear:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"MAR: {mar:.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(processed_frame, f"Status: {status}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            except Exception as e:
                print(f"Error processing landmarks: {e}")
                status = "Landmark error"
        
        return drowsy, drowsy_prob, processed_frame, ear, mar, status

class LBFFaceLandmarkDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("lbfmodel.yaml")  # Make sure this file is in your project
        
        # Constants
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.6
        self.LEFT_EYE_INDICES = list(range(36, 42))
        self.RIGHT_EYE_INDICES = list(range(42, 48))
        self.MOUTH_INDICES = list(range(48, 68))
        
    def calculate_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def calculate_mar(self, mouth):
        A = distance.euclidean(mouth[2], mouth[10])
        B = distance.euclidean(mouth[3], mouth[9])
        C = distance.euclidean(mouth[4], mouth[8])
        D = distance.euclidean(mouth[0], mouth[6])
        return (A + B + C) / (3.0 * D)
    
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
        )
        
        result_image = image.copy()
        status = "No face detected"
        ear = 0.0
        mar = 0.0
        drowsy_prob = 0.0
        
        if len(faces) > 0:
            ok, landmarks = self.facemark.fit(gray, faces)
            
            if ok:
                face_landmarks = np.array(landmarks[0][0], dtype=np.int32)
                
                # Extract features
                left_eye = face_landmarks[self.LEFT_EYE_INDICES]
                right_eye = face_landmarks[self.RIGHT_EYE_INDICES]
                mouth = face_landmarks[self.MOUTH_INDICES]
                
                # Calculate metrics
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(mouth)
                
                # Determine status
                if ear < self.EAR_THRESHOLD and mar > self.MAR_THRESHOLD:
                    status = "Drowsy + Yawning"
                    drowsy_prob = 0.9
                elif ear < self.EAR_THRESHOLD:
                    status = "Drowsy"
                    drowsy_prob = 0.7
                elif mar > self.MAR_THRESHOLD:
                    status = "Yawning"
                    drowsy_prob = 0.6
                else:
                    status = "Alert"
                    drowsy_prob = 0.1
                
                # Draw landmarks
                for point in face_landmarks:
                    cv2.circle(result_image, tuple(point), 1, (0, 255, 0), -1)
                for point in left_eye:
                    cv2.circle(result_image, tuple(point), 1, (255, 0, 0), -1)
                for point in right_eye:
                    cv2.circle(result_image, tuple(point), 1, (255, 0, 0), -1)
                for point in mouth:
                    cv2.circle(result_image, tuple(point), 1, (0, 0, 255), -1)
                
                # Add text
                cv2.putText(result_image, f"Status: {status}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_image, f"EAR: {ear:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(result_image, f"MAR: {mar:.2f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                cv2.putText(result_image, "Landmarks not detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return drowsy_prob, status, result_image, ear, mar

class MediaPipeImageDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.EAR_THRESHOLD = 0.24
        self.MAR_THRESHOLD = 0.6
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.OUTER_MOUTH = [61, 81, 311, 308, 402, 314, 317, 87]

    def calculate_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth):
        A = distance.euclidean(mouth[1], mouth[7])
        B = distance.euclidean(mouth[2], mouth[6])
        C = distance.euclidean(mouth[3], mouth[5])
        D = distance.euclidean(mouth[0], mouth[4])
        return (A + B + C) / (3.0 * D)

    def detect(self, image):
        h, w = image.shape[:2]
        results = self.face_mesh.process(image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            def get_coords(indices):
                return [(int(landmarks.landmark[i].x * w), 
                        int(landmarks.landmark[i].y * h)) for i in indices]
            
            left_eye = get_coords(self.LEFT_EYE)
            right_eye = get_coords(self.RIGHT_EYE)
            mouth = get_coords(self.OUTER_MOUTH)
            
            ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
            mar = self.calculate_mar(mouth)
            
            status = "Alert"
            if ear < self.EAR_THRESHOLD and mar > self.MAR_THRESHOLD:
                status = "Drowsy + Yawning"
                drowsy_prob = 0.9
            elif ear < self.EAR_THRESHOLD:
                status = "Drowsy"
                drowsy_prob = 0.7
            elif mar > self.MAR_THRESHOLD:
                status = "Yawning"
                drowsy_prob = 0.6
            else:
                drowsy_prob = 0.1
                
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            
            for x, y in left_eye + right_eye:
                ax.plot(x, y, 'go', markersize=5)
            for x, y in mouth:
                ax.plot(x, y, 'mo', markersize=5)
                
            ax.text(10, 30, f"EAR: {ear:.2f}", color='yellow', fontsize=12,
                   bbox=dict(facecolor='black', alpha=0.8))
            ax.text(10, 60, f"MAR: {mar:.2f}", color='magenta', fontsize=12,
                   bbox=dict(facecolor='black', alpha=0.8))
            ax.text(10, 90, f"Status: {status}", color='red', fontsize=14,
                   bbox=dict(facecolor='black', alpha=0.8))
            
            ax.axis('off')
            fig.tight_layout()
            
            return drowsy_prob, status, fig, ear, mar
        else:
            return 0.0, "No face detected", None, 0.0, 0.0

# Load model
@st.cache_resource
def load_model(model_path, model_type):
    if model_type == "Blur Detection":
        return BlurDetector(), None
    elif model_type == "Realtime Detection (Mediapipe)":
        return MediaPipeDrowsinessDetector(), None
    elif model_type == "Detection /w image via LBF":
        return LBFFaceLandmarkDetector(), None
    elif model_type == "Detection /w image via Mediapipe":
        return MediaPipeImageDetector(), None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "CNN-LSTM":
        model = CNN_LSTM(num_classes=2)
    else:  # model_type == "CNN"
        model = DrowsinessCNN()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define image preprocessing
def preprocess_frame(frame, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(frame)

# Function to make prediction with CNN-LSTM model
def predict_drowsiness_cnn_lstm(model, device, frame_window):
    if len(frame_window) != 16:
        return None, 0.0
    
    # Stack frames into a tensor of shape [1, 16, 3, 224, 224]
    frames_tensor = torch.stack(frame_window).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(frames_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        drowsy_prob = probs[0, 1].item()
    
    return preds.item(), drowsy_prob

# Function to make prediction with CNN model
def predict_drowsiness_cnn(model, device, frame):
    # Process single frame
    frame_tensor = frame.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(frame_tensor)
        drowsy_prob = output.item()
        pred = 1 if drowsy_prob >= 0.5 else 0
    
    return pred, drowsy_prob

# Main function
def main():
    st.title("Real-time Drowsiness Detection System")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_type = st.sidebar.selectbox("Select Model", ["CNN-LSTM", "CNN", "Blur Detection", "Realtime Detection (Mediapipe)",
                                                       "Detection /w image via LBF", "Detection /w image via Mediapipe"])
    
    # Model paths
    model_paths = {
        "CNN-LSTM": "./cnn_lstm_test2.pth",
        "CNN": "./best_model.pth",
        "Blur Detection": None,
        "Realtime Detection (Mediapipe)": None,
        "Detection /w image via LBF": None,
        "Detection /w image via Mediapipe": None
    }
    
    # Model-specific settings
    if model_type == "CNN-LSTM":
        target_size = (224, 224)
        window_size = 16
    elif model_type == "CNN":
        target_size = (112, 112)
        window_size = 1
    elif model_type == "Realtime Detection (Mediapipe)":
        target_size = (640, 480)
        window_size = 1
    else:  # Blur Detection
        target_size = (640, 480)  # Higher resolution for better blur detection
        window_size = 1

    
    # Blur detection parameters (only visible when Blur Detection is selected)
    if model_type == "Blur Detection":
        st.sidebar.subheader("Blur Detection Settings")
        laplacian_threshold = st.sidebar.slider("Laplacian Threshold", 50, 200, 100, 5)
        tenengrad_threshold = st.sidebar.slider("Tenengrad Threshold", 10000, 30000, 20000, 1000)
        sobel_threshold = st.sidebar.slider("Sobel Threshold", 5, 30, 15, 1)
    
    # Alert settings
    # st.sidebar.subheader("Alert Settings")
    # alert_threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.01)
    if model_type != "Realtime Detection (Mediapipe)" and model_type != "Detection /w image via LBF":
        st.sidebar.subheader("Alert Settings")
        alert_threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.01)
    else:
        # Set a default threshold for MediaPipe
        alert_threshold = 0.5

    
    # Video capture settings
    st.sidebar.subheader("Video Settings")
    video_source = st.sidebar.selectbox("Video Source", ["Webcam", "Upload Video File"])
    
    # For file upload
    uploaded_file = None
    if video_source == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov"])
    
    # Load model
    model_loaded = load_model(model_paths[model_type], model_type)
    if not model_loaded:
        st.error(f"Failed to load {model_type} model. Please check the model path.")
        return
    
    model, device = model_loaded
    
    # Set blur detector parameters if that's the selected model
    if model_type == "Blur Detection" and isinstance(model, BlurDetector):
        model.laplacian_threshold = laplacian_threshold
        model.tenengrad_threshold = tenengrad_threshold
        model.sobel_threshold = sobel_threshold
    
    # Initialize session state for control flow
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start Detection", key="start_button", disabled=st.session_state.detection_running):
            st.session_state.detection_running = True
            st.rerun()
    
    with col2:
        if st.button("Stop Detection", key="stop_button", disabled=not st.session_state.detection_running):
            st.session_state.detection_running = False
            st.rerun()
    
    # Model info
    st.sidebar.markdown(f"### Current Model: {model_type}")
    if model_type == "CNN-LSTM":
        st.sidebar.info("This model analyzes sequences of 16 frames to detect drowsiness patterns over time.")
    elif model_type == "CNN":
        st.sidebar.info("This model analyzes individual frames to detect drowsiness based on single images.")
    elif model_type == "Realtime Detection (Mediapipe)":
        st.sidebar.info("This model analyzes individual frames to detect EAR and MAR to determine if the person is drowsy. You must use the webcam for this")
    elif model_type == "Detection /w image via LBF":
        st.sidebar.info("This model analyzes individual frames to detect EAR and MAR to determine if the person is drowsy. You must upload an image for this.")
    elif model_type == "Detection /w image via Mediapipe":
        st.sidebar.info("This model analyzes individual frames to detect EAR and MAR to determine if the person is drowsy. You must upload an image for this.")
    else:  # Blur Detection
        st.sidebar.info("This model detects drowsiness by analyzing frame blurriness, which can indicate eye closure or nodding. You must use webcam for this.")
    
    # Initialize UI containers
    stframe = st.empty()
    status_text = st.empty()
    alert_placeholder = st.empty()
    
    # Processing loop - only runs when detection is active
    if st.session_state.detection_running:
        # Set up video capture
        cap = None
        if video_source == "Webcam":
            cap = cv2.VideoCapture(0)
        elif uploaded_file is not None:
            # Save the uploaded file to disk
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_file)
        
        if cap is None or not cap.isOpened():
            st.error("Failed to open video source.")
            st.session_state.detection_running = False
            return
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            drowsy_gauge = st.empty()
        with col2:
            alert_time = st.empty()
            drowsy_time = st.empty()
        
        # Initialize metrics
        total_time = 0
        drowsy_duration = 0
        alert_duration = 0
        last_time = time.time()
        drowsy_status = False
        
        # Initialize frame window for CNN-LSTM model
        frame_window = deque(maxlen=window_size)
        
        # For blur detection, add a container for blur metrics
        if model_type == "Blur Detection":
            blur_metrics = st.empty()
        
        # Main detection loop
        try:
            while st.session_state.detection_running:
                ret, frame = cap.read()
                
                if not ret:
                    if video_source == "Upload Video File":
                        st.info("Video ended. Restarting...")
                        cap = cv2.VideoCapture(temp_file)
                        continue
                    else:
                        st.error("Failed to capture frame.")
                        break
                
                # Resize frame according to model requirements
                frame = cv2.resize(frame, target_size)
                
                # Convert to RGB for display and processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Different processing based on model type
                if model_type == "Realtime Detection (Mediapipe)":
                    prediction, drowsy_prob, processed_frame, ear, mar, status = model.detect(frame)
                    
                    # Update metrics
                    current_time = time.time()
                    delta_time = current_time - last_time
                    total_time += delta_time
                    
                    if prediction == 1:  # Drowsy
                        drowsy_duration += delta_time
                        status_color = "red"
                    else:  # Alert
                        alert_duration += delta_time
                        status_color = "green"
                    
                    last_time = current_time
                    
                    # Display status
                    status_text.markdown(f"<h2 style='color: {status_color};'>{status}</h2>", unsafe_allow_html=True)
                    
                    # Display alert if threshold exceeded
                    if drowsy_prob > alert_threshold:
                        alert_placeholder.warning("WARNING: Drowsiness detected! Please take a break.")
                    else:
                        alert_placeholder.empty()
                    
                    # Update metrics display
                    drowsy_gauge.progress(drowsy_prob)
                    alert_time.metric("Alert Time", f"{alert_duration:.1f}s", f"{(alert_duration/total_time)*100:.1f}%")
                    drowsy_time.metric("Drowsy Time", f"{drowsy_duration:.1f}s", f"{(drowsy_duration/total_time)*100:.1f}%")
                    
                    # Display the processed frame with landmarks
                    stframe.image(processed_frame, channels="BGR", use_column_width=True)
                    continue  # Skip the rest of the loop for this iteration

                elif model_type == "Blur Detection":
                    # Use blur detection directly on the frame
                    prediction, drowsy_prob, blur_scores = model.detect_drowsiness(frame_rgb)
                    
                    # Display blur metrics
                    blur_metrics.markdown(f"""
                    ### Blur Metrics
                    - Laplacian: {blur_scores['laplacian']:.1f} (Threshold: {model.laplacian_threshold})
                    - Tenengrad: {blur_scores['tenengrad']:.1f} (Threshold: {model.tenengrad_threshold})
                    - Sobel: {blur_scores['sobel']:.1f} (Threshold: {model.sobel_threshold})
                    """)
                
                    
                else:
                    # For CNN and CNN-LSTM, preprocess the frame
                    processed_frame = preprocess_frame(frame_rgb, target_size=target_size)
                    
                    # Add to frame window
                    frame_window.append(processed_frame)
                    
                    # Make prediction based on model type
                    if model_type == "CNN-LSTM" and len(frame_window) == window_size:
                        prediction, drowsy_prob = predict_drowsiness_cnn_lstm(model, device, list(frame_window))
                    elif model_type == "CNN":
                        prediction, drowsy_prob = predict_drowsiness_cnn(model, device, processed_frame)
                    else:
                        # Still collecting frames for CNN-LSTM
                        prediction, drowsy_prob = None, 0.0
                
                # Update metrics if we have a prediction
                if prediction is not None:
                    # Update metrics
                    current_time = time.time()
                    delta_time = current_time - last_time
                    total_time += delta_time
                    
                    if prediction == 1:  # Drowsy
                        drowsy_duration += delta_time
                        drowsy_status = True
                        status = "⚠️ DROWSY"
                        status_color = "red"
                    else:  # Alert
                        alert_duration += delta_time
                        drowsy_status = False
                        status = "✅ ALERT"
                        status_color = "green"
                    
                    last_time = current_time
                    
                    # Display status
                    status_text.markdown(f"<h2 style='color: {status_color};'>{status}</h2>", unsafe_allow_html=True)
                    
                    # Display alert if threshold exceeded
                    if drowsy_prob > alert_threshold:
                        alert_placeholder.warning("WARNING: Drowsiness detected! Please take a break.")
                    else:
                        alert_placeholder.empty()
                    
                    # Update metrics display
                    drowsy_gauge.progress(drowsy_prob)
                    alert_time.metric("Alert Time", f"{alert_duration:.1f}s", f"{(alert_duration/total_time)*100:.1f}%")
                    drowsy_time.metric("Drowsy Time", f"{drowsy_duration:.1f}s", f"{(drowsy_duration/total_time)*100:.1f}%")
                    
                    # Draw status on frame
                    cv2.putText(
                        frame_rgb, 
                        f"Status: {status}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0) if not drowsy_status else (0, 0, 255), 
                        2
                    )
                    cv2.putText(
                        frame_rgb, 
                        f"Drowsy Prob: {drowsy_prob:.2f}", 
                        (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0) if not drowsy_status else (0, 0, 255), 
                        2
                    )
                elif model_type == "CNN-LSTM":
                    # Still collecting frames for CNN-LSTM
                    remaining = window_size - len(frame_window)
                    status_text.info(f"Collecting frames... {remaining} more needed")
                
                # Display the frame
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Check if we should stop (by checking the session state)
                if not st.session_state.detection_running:
                    break
                
                # Add a small delay to prevent high CPU usage and give time for button clicks
                time.sleep(0.01)
                
        finally:
            if cap is not None:
                cap.release()
            if video_source == "Upload Video File" and uploaded_file is not None:
                if os.path.exists("temp_video.mp4"):
                    os.remove("temp_video.mp4")

    elif model_type == "Detection /w image via LBF":
        with st.sidebar:
            st.subheader("Image Settings")
            uploaded_file = st.file_uploader(
                "Upload face image", 
                type=["jpg", "jpeg", "png"],
                key="lbf_uploader"
            )
            
            # Optional: Add threshold controls
            st.subheader("Detection Thresholds")
            ear_threshold = st.slider("EAR Threshold", 0.1, 0.5, 0.25, 0.01,
                                    help="Eye Aspect Ratio threshold (lower = more sensitive)")
            mar_threshold = st.slider("MAR Threshold", 0.3, 1.0, 0.6, 0.05,
                                    help="Mouth Aspect Ratio threshold (higher = more sensitive)")
        
        # Main area for results
        if uploaded_file is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process the image
            model, _ = load_model(None, model_type)
            
            # Update model thresholds from sidebar
            model.EAR_THRESHOLD = ear_threshold
            model.MAR_THRESHOLD = mar_threshold
            
            drowsy_prob, status, result_image, ear, mar = model.detect(image)
            
            # Display results in main area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(result_image, channels="BGR", use_container_width=True, 
                    caption="Processed Image with Landmarks")
            
            with col2:
                st.subheader("Detection Results")
                st.metric("Eye Aspect Ratio (EAR)", f"{ear:.2f}", 
                        delta=f"Threshold: {ear_threshold:.2f}")
                st.metric("Mouth Aspect Ratio (MAR)", f"{mar:.2f}",
                        delta=f"Threshold: {mar_threshold:.2f}")
                st.metric("Status", status)
                
                if "Drowsy" in status:
                    st.error("Drowsiness detected!")
                elif "Yawning" in status:
                    st.warning("Yawning detected!")
                else:
                    st.success("Alert state detected")
    
    if model_type == "Detection /w image via Mediapipe":
        with st.sidebar:
            st.subheader("Image Settings")
            uploaded_file = st.file_uploader(
                "Upload face image", 
                type=["jpg", "jpeg", "png"],
                key="lbf_uploader"
            )
            
            # Optional: Add threshold controls
            st.subheader("Detection Thresholds")
            ear_threshold = st.slider("EAR Threshold", 0.1, 0.5, 0.25, 0.01,
                                    help="Eye Aspect Ratio threshold (lower = more sensitive)")
            mar_threshold = st.slider("MAR Threshold", 0.3, 1.0, 0.6, 0.05,
                                    help="Mouth Aspect Ratio threshold (higher = more sensitive)")

        # st.header("Detection /w image via Mediapipe")
        # if model_type == "MediaPipe Image Detection":
        #     with st.sidebar:
        #         st.subheader("Detection Thresholds")
        #         ear_thresh = st.slider("EAR Threshold", 0.1, 0.5, 0.24, 0.01)
        #         mar_thresh = st.slider("MAR Threshold", 0.3, 1.0, 0.6, 0.05)
        #         # Update model thresholds
        #         model.EAR_THRESHOLD = ear_thresh
        #         model.MAR_THRESHOLD = mar_thresh
                
        # # if st.sidebar.button("Use Sample Image"):
        # #     # Include sample images in your app directory
        # #     sample_images = ["sample1.jpg", "sample2.jpg"]
        # #     selected = np.random.choice(sample_images)
        # #     image = plt.imread(selected)
        # #     # Process the sample image...

    
        # uploaded_file = st.file_uploader(
        #     "Upload an image", 
        #     type=["jpg", "jpeg", "png"],
        #     key="mediapipe_uploader"
        # )
        
        if uploaded_file is not None:
            # Read image
            image = plt.imread(uploaded_file)
            
            # Process image
            model, _ = load_model(None, model_type)
            model.EAR_THRESHOLD = ear_threshold
            model.MAR_THRESHOLD = mar_threshold
            drowsy_prob, status, fig, ear, mar = model.detect(image)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.image(image, use_container_width=True)
                    st.warning("No face detected in the image!")
            
            with col2:
                st.subheader("Detection Results")
                st.metric("EAR (Eye Aspect Ratio)", f"{ear:.2f}", 
                        help="Values below 0.24 suggest drowsiness")
                st.metric("MAR (Mouth Aspect Ratio)", f"{mar:.2f}",
                        help="Values above 0.6 suggest yawning")
                st.metric("Status", status)
                
                if "Drowsy" in status:
                    st.error("Drowsiness detected!")
                elif "Yawning" in status:
                    st.warning("Yawning detected!")
                else:
                    st.success("Alert state detected")
        
    else:
        # Display a placeholder when not active
        st.info("Click 'Start Detection' to begin drowsiness monitoring.")
        
        # Display model comparison info
        st.subheader("About the Models")
        
        col1, col2, col3 = st.columns(3)
 
        with col1:
            st.markdown("### CNN-LSTM Model")
            st.markdown("""
            - **Approach**: Temporal sequence analysis
            - **Input**: Sequence of 16 video frames
            - **Strength**: Captures patterns over time
            - **Best for**: Detecting gradual drowsiness onset
            - **Processing**: Slower but potentially more accurate
            """)
        
        with col2:
            st.markdown("### CNN Model")
            st.markdown("""
            - **Approach**: Single-frame analysis
            - **Input**: Individual video frames
            - **Strength**: Fast, immediate analysis
            - **Best for**: Detecting current drowsiness state
            - **Processing**: Faster but may miss temporal patterns
            """)
            
        with col3:
            st.markdown("### Blur Detection")
            st.markdown("""
            - **Approach**: Motion blur analysis
            - **Input**: Raw video frames
            - **Strength**: No ML model required, works on any camera
            - **Best for**: Low-resource environments
            - **Processing**: Very fast, detects nodding and eye closure
            """)

        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("### Realtime Detection with Mediapipe")
            st.markdown("""
            - **Approach**: Facial landmarks analysis
            - **Input**: Raw video frames through webcam
            - **Strength**: Low latency, lightweight model suitable for real-time applications
            - **Best for**: Continuous monitoring of driver/operator alertness
            - **Processing**: Calculates EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) to detect eye blinks, prolonged eye closure, and yawning
            """)


        with col5:
            st.write("### Detection with Images via LBF and dlib")
            st.markdown("""
            - **Approach**: Facial landmarks analysis with LBF model/dlib
            - **Input**: Image frames (static)
            - **Strength**: Faster than deep learning models, works well on low-resource devices
            - **Best for**: Offline processing or systems with limited computational power
            - **Processing**: Detects 68 facial landmarks, computes EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) for blink/drowsiness detection
            """)

        with col6:
            st.write("### Detection with Videos Frames")
            st.markdown("""
            - **Approach**: dlib's 68-point facial landmarks + temporal feature engineering
            - **Input**: Sequential video frames (recorded)
            - **Strength**: Robust due to time-series analysis, hand-engineered features (**PERCLOS, BF, OV, CV, MCD, AOL**) improve interpretability  
            - **Best for**: Long-term drowsiness monitoring
            - **Processing**: Very fast with promising results
            """)


if __name__ == "__main__":
    main()