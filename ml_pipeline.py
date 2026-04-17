import os
import cv2
import numpy as np
import tensorflow as tf
import base64
from insightface.app import FaceAnalysis

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

ENSEMBLE_WEIGHTS = {'xception': 0.45, 'capsnet': 0.30, 'mesonet': 0.20}
CLASSIFICATION_THRESHOLD = 0.46
FRAMES_PER_VIDEO = 100
BLUR_THRESHOLD = 30

# ==========================================
# 2. MODEL LOADING
# ==========================================

# @tf.keras.saving.register_keras_serializable()
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())

# @tf.keras.saving.register_keras_serializable()
class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, **kwargs): 
        super().__init__(**kwargs) 
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.num_capsules * self.dim_capsules),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (batch_size, 784, 256)
        u_hat = tf.matmul(inputs, self.W)
        
        # Reshape exactly as done in your training notebook
        # Output shape becomes: (batch_size, 784, 4, 8)
        u_hat = tf.reshape(u_hat, (-1, tf.shape(inputs)[1], self.num_capsules, self.dim_capsules))
        
        # Apply your custom squash function
        return squash(u_hat)

    def compute_output_shape(self, input_shape):
        # This tells Keras to expect 784 * 4 * 8 = 25088 when flattened
        return (input_shape[0], input_shape[1], self.num_capsules, self.dim_capsules)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules
        })
        return config

# Load your fine-tuned, robust models
mesonet = tf.keras.models.load_model('models/mesonet_robust.keras')
xception = tf.keras.models.load_model('models/xception_robust.keras')
capsnet = tf.keras.models.load_model('models/capsnet_final.keras', custom_objects={'squash': squash, 'CapsuleLayer': CapsuleLayer})

# ==========================================
# 3. PREPROCESSING
# ==========================================

def is_blurry(img, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold

def extract_primary_face(frame):
    faces = app.get(frame)
    if len(faces) == 0:
        return None, None, None

    face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
    x1, y1, x2, y2 = map(int, face.bbox)
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0 or is_blurry(crop):
        return None, None, None

    # Keep a display-sized BGR crop for the UI
    display_crop = cv2.resize(crop, (150, 150))

    # Convert to RGB for models
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 224x224 scaled to [-1, 1] for Xception (and MesoNet if trained with same pipeline)
    crop_224 = tf.keras.applications.xception.preprocess_input(cv2.resize(crop_rgb, (224, 224)).astype('float32'))
    
    # 128x128 scaled to [0, 1] for CapsNet
    crop_128 = cv2.resize(crop_rgb, (128, 128)).astype('float32') / 255.0
    
    return np.expand_dims(crop_224, axis=0), np.expand_dims(crop_128, axis=0), display_crop

def predict_frame(img_224, img_128):
    pred_meso = float(mesonet.predict(img_224, verbose=0)[0][0])
    pred_xcep = float(xception.predict(img_224, verbose=0)[0][0])
    pred_caps = float(capsnet.predict(img_128, verbose=0)[0][0])
    
    ensemble = (
        (pred_xcep * ENSEMBLE_WEIGHTS['xception']) +
        (pred_caps * ENSEMBLE_WEIGHTS['capsnet']) +
        (pred_meso * ENSEMBLE_WEIGHTS['mesonet'])
    )
    
    return {'mesonet': pred_meso, 'xception': pred_xcep, 'capsnet': pred_caps, 'ensemble': ensemble}

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def process_media(file_path, is_video=False):
    extracted_display_frames = []

    if not is_video:
        img = cv2.imread(file_path)
        img_224, img_128, display_crop = extract_primary_face(img)
        if img_224 is None:
            return {"error": "No clear/valid face detected in image."}
            
        scores = predict_frame(img_224, img_128)
        scores['label'] = "Fake" if scores['ensemble'] >= CLASSIFICATION_THRESHOLD else "Real"
        scores['frames_analyzed'] = 1
        
        _, buffer = cv2.imencode('.jpg', display_crop)
        scores['frames_base64'] = [base64.b64encode(buffer).decode('utf-8')]
        return scores

    cap = cv2.VideoCapture(file_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: return {"error": "Unreadable video."}

    step = max(1, total // (FRAMES_PER_VIDEO * 2))
    aggregated_scores = {'mesonet': [], 'xception': [], 'capsnet': [], 'ensemble': []}
    saved_count = 0
    idx = 0

    while cap.isOpened() and saved_count < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret: break

        if idx % step == 0:
            img_224, img_128, display_crop = extract_primary_face(frame)
            if img_224 is not None:
                scores = predict_frame(img_224, img_128)
                for k in aggregated_scores.keys():
                    aggregated_scores[k].append(scores[k])
                extracted_display_frames.append(display_crop)
                saved_count += 1
        idx += 1
    cap.release()

    if saved_count == 0:
        return {"error": "Could not extract valid faces from the video."}

    final_results = {k: float(np.mean(v)) for k, v in aggregated_scores.items()}
    final_results['label'] = "Real" if final_results['ensemble'] >= CLASSIFICATION_THRESHOLD else "Fake"
    final_results['frames_analyzed'] = saved_count
    
    # Sample up to 6 evenly spaced frames for the UI
    sample_rate = max(1, len(extracted_display_frames) // 6)
    sampled_frames = extracted_display_frames[::sample_rate][:6]
    
    encoded_frames = []
    for f in sampled_frames:
        _, buffer = cv2.imencode('.jpg', f)
        encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))
        
    final_results['frames_base64'] = encoded_frames
    
    return final_results