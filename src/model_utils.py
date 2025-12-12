"""
Model Utilities for AI MoodSense
Uses DeepFace for emotion recognition with correct 3-class mapping
"""

import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image
import os
import tempfile

# Label names
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']

# Emoji mapping
EMOJI_MAP = {
    'Positive': '😊',
    'Neutral': '😐',
    'Negative': '😞'
}

# DeepFace emotion to our 3-class mapping
# DeepFace returns: angry, disgust, fear, happy, sad, surprise, neutral
DEEPFACE_TO_CLASS = {
    'angry': 'Negative',
    'disgust': 'Negative',
    'fear': 'Negative',
    'sad': 'Negative',
    'happy': 'Positive',
    'surprise': 'Positive',
    'neutral': 'Neutral'
}

class EmotionPredictor:
    """Class for emotion prediction using DeepFace"""
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion predictor using DeepFace
        
        Args:
            model_path: Not used for DeepFace (kept for compatibility)
        """
        self.model_path = model_path
        self.model = "DeepFace"  # For compatibility
        print("Using DeepFace for emotion recognition")
        print("DeepFace model loaded successfully")
    
    def load_model(self):
        """DeepFace doesn't need explicit loading"""
        pass
    
    def preprocess_image(self, image):
        """
        Preprocess image for DeepFace (saves to temp file if needed)
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Path to image file (DeepFace requires file path)
        """
        # If already a file path, return it
        if isinstance(image, str):
            if os.path.exists(image):
                return image
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert numpy array to BGR if needed (OpenCV format)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save to temporary file for DeepFace
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, image)
        return temp_file.name
    
    def detect_face(self, image):
        """
        Detect face in image using OpenCV (optional, DeepFace can also detect)
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            face_image: Cropped face image, or None if no face detected
            face_coords: (x, y, w, h) coordinates of detected face
        """
        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Use the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Crop face
        face_image = image[y:y+h, x:x+w]
        
        return face_image, (x, y, w, h)

    def detect_faces(self, image, max_faces=5):
        """
        Detect multiple faces in an image using OpenCV Haar cascades.
        
        Args:
            image: Input image (numpy array or file path)
            max_faces: Maximum number of faces to return (sorted by size)
            
        Returns:
            List of tuples: (cropped_face_image, (x, y, w, h))
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Sort faces by area (largest first) and limit
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        results = []
        for (x, y, w, h) in faces:
            cropped = image[y:y+h, x:x+w]
            results.append((cropped, (x, y, w, h)))
        return results
    
    def predict(self, image, detect_face=True, temperature=1.2, apply_bias_correction=False, debug=False):
        """
        Predict emotion from image using DeepFace
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            detect_face: Whether to detect and crop face first (optional)
            temperature: Not used for DeepFace (kept for compatibility)
            apply_bias_correction: Not used for DeepFace (kept for compatibility)
            debug: Print debug information
            
        Returns:
            prediction: Dictionary with 'emotion', 'confidence', 'probabilities'
        """
        try:
            # Detect and crop face if requested
            if detect_face:
                face_image, face_coords = self.detect_face(image)
                if face_image is None:
                    return {
                        'emotion': 'No Face Detected',
                        'confidence': 0.0,
                        'probabilities': {},
                        'error': 'No face detected in the image'
                    }
                image = face_image
            
            # Preprocess image (save to temp file for DeepFace)
            image_path = self.preprocess_image(image)
            
            if debug:
                print(f"Analyzing image: {image_path}")
            
            # Use DeepFace to analyze emotions with improved settings
            # Try multiple backends for better accuracy (start with strongest)
            backends = ['retinaface', 'mtcnn', 'ssd', 'dlib', 'opencv']
            result = None
            last_error = None
            
            # Try different backends until one works
            for backend in backends:
                try:
                    result = DeepFace.analyze(
                        img_path=image_path,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend=backend,
                        silent=True
                    )
                    if debug:
                        print(f"Successfully used backend: {backend}")
                    break
                except Exception as e:
                    last_error = e
                    if debug:
                        print(f"Backend {backend} failed: {str(e)}")
                    continue
            
            if result is None:
                raise Exception(f"All backends failed. Last error: {str(last_error)}")
            
            # Clean up temp file
            try:
                os.unlink(image_path)
            except:
                pass
            
            # DeepFace returns a list, get first result
            if isinstance(result, list):
                emotions = result[0].get('emotion', {})
            else:
                emotions = result.get('emotion', {})
            
            if debug:
                print(f"DeepFace raw emotions: {emotions}")
            
            # Map DeepFace emotions to our 3 classes with balanced weighting
            class_probs = {
                'Negative': 0.0,
                'Neutral': 0.0,
                'Positive': 0.0
            }
            
            # Get raw emotion probabilities
            raw_emotions = {k.lower(): v for k, v in emotions.items()}
            
            # Balanced emotion weights - strongly favor positive/negative over neutral
            emotion_weights = {
                'angry': 1.1,      # Slight boost for negative
                'disgust': 1.05,
                'fear': 0.85,      # Further downweight fear vs surprise to avoid false negatives
                'sad': 1.15,       # Boost sad to catch mild sadness
                'neutral': 0.85,   # Reduce neutral weight - make it less dominant
                'happy': 1.4,      # Strong boost for positive emotions
                'surprise': 2.25   # Extra boost for surprise ("wow" expressions)
            }
            
            # Sum weighted probabilities for each class
            for emotion, prob in raw_emotions.items():
                if emotion in DEEPFACE_TO_CLASS:
                    class_name = DEEPFACE_TO_CLASS[emotion]
                    weight = emotion_weights.get(emotion, 1.0)
                    class_probs[class_name] += (prob / 100.0) * weight
            
            # Normalize probabilities
            total = sum(class_probs.values())
            if total > 0:
                class_probs = {k: v / total for k, v in class_probs.items()}
            else:
                # Fallback if no emotions detected
                class_probs = {'Negative': 0.33, 'Neutral': 0.33, 'Positive': 0.34}
            
            # Get raw emotion values for threshold checks
            negative_emotions = ['angry', 'disgust', 'fear', 'sad']
            positive_emotions = ['happy', 'surprise']
            max_negative_raw = max([raw_emotions.get(em, 0) for em in negative_emotions])
            max_positive_raw = max([raw_emotions.get(em, 0) for em in positive_emotions])
            neutral_raw = raw_emotions.get('neutral', 0)
            
            # PRIORITY 1: If positive emotions are present (even very mildly), strongly favor positive over neutral
            # Special handling for surprise - very sensitive threshold
            surprise_raw = raw_emotions.get('surprise', 0)
            happy_raw = raw_emotions.get('happy', 0)
            
            # Pre-boost for mixed happy+fear "wow" faces where surprise may be zeroed out
            happy_raw = raw_emotions.get('happy', 0)
            fear_raw = raw_emotions.get('fear', 0)
            if raw_emotions.get('surprise', 0) < 5.0 and happy_raw > 20.0 and fear_raw > 20.0:
                # Nudge probability toward Positive to avoid fear dominance on open-mouth surprise
                boost = 0.1
                neg_reduction = min(class_probs['Negative'], boost * 0.6)
                neu_reduction = min(class_probs['Neutral'], boost * 0.4)
                class_probs['Negative'] -= neg_reduction
                class_probs['Neutral'] -= neu_reduction
                class_probs['Positive'] += (neg_reduction + neu_reduction)
                total = sum(class_probs.values())
                class_probs = {k: v / total for k, v in class_probs.items()}

            if max_positive_raw > 15.0:  # Very low threshold - catch even mild positive expressions
                if class_probs['Positive'] < class_probs['Neutral'] or class_probs['Positive'] < class_probs['Negative']:
                    # Positive emotions detected but not winning - strongly boost positive
                    if surprise_raw > 12.0:  # Even very mild surprise should be recognized
                        # Surprise detected - strongly favor positive
                        if class_probs['Positive'] < class_probs['Neutral']:
                            boost = (class_probs['Neutral'] - class_probs['Positive']) * 0.7  # Strong boost
                            class_probs['Positive'] = class_probs['Positive'] + boost
                            class_probs['Neutral'] = class_probs['Neutral'] - boost
                        if class_probs['Positive'] < class_probs['Negative']:
                            boost = (class_probs['Negative'] - class_probs['Positive']) * 0.6
                            class_probs['Positive'] = class_probs['Positive'] + boost
                            class_probs['Negative'] = class_probs['Negative'] - boost
                        # Renormalize
                        total = sum(class_probs.values())
                        class_probs = {k: v / total for k, v in class_probs.items()}
                    elif happy_raw > 15.0 or max_positive_raw > neutral_raw:
                        # Happy or positive emotions are stronger than neutral - boost positive
                        if class_probs['Positive'] < class_probs['Neutral']:
                            boost = (class_probs['Neutral'] - class_probs['Positive']) * 0.6
                            class_probs['Positive'] = class_probs['Positive'] + boost
                            class_probs['Neutral'] = class_probs['Neutral'] - boost
                            # Renormalize
                            total = sum(class_probs.values())
                            class_probs = {k: v / total for k, v in class_probs.items()}
            
            # PRIORITY 2: If negative emotions are present (even very mildly), strongly favor negative over neutral
            sad_raw = raw_emotions.get('sad', 0)
            
            if max_negative_raw > 18.0:  # Very low threshold - catch even mild sadness
                if class_probs['Negative'] < class_probs['Neutral']:
                    # Negative emotions detected but neutral is winning - strongly boost negative
                    if sad_raw > 15.0:  # Even mild sadness should be recognized
                        # Sadness detected - strongly favor negative
                        boost = (class_probs['Neutral'] - class_probs['Negative']) * 0.7  # Strong boost
                        class_probs['Negative'] = class_probs['Negative'] + boost
                        class_probs['Neutral'] = class_probs['Neutral'] - boost
                        # Renormalize
                        total = sum(class_probs.values())
                        class_probs = {k: v / total for k, v in class_probs.items()}
                    elif max_negative_raw > neutral_raw:
                        # Negative raw score is higher than neutral - negative should win
                        boost = (class_probs['Neutral'] - class_probs['Negative']) * 0.6
                        class_probs['Negative'] = class_probs['Negative'] + boost
                        class_probs['Neutral'] = class_probs['Neutral'] - boost
                        # Renormalize
                        total = sum(class_probs.values())
                        class_probs = {k: v / total for k, v in class_probs.items()}
            
            # PRIORITY 3: Aggressively reduce neutral when any emotion signals are present
            # If neutral is winning but positive/negative have ANY signals, strongly reduce neutral
            if class_probs['Neutral'] > 0.45:  # Neutral is dominant or near-dominant
                if max_positive_raw > 12.0 or max_negative_raw > 15.0:
                    # There are emotion signals - strongly reduce neutral dominance
                    reduction = class_probs['Neutral'] * 0.3  # Reduce neutral by 30%
                    class_probs['Neutral'] = class_probs['Neutral'] - reduction
                    
                    # Distribute to the stronger signal (favor the one with higher raw score)
                    if max_positive_raw > max_negative_raw:
                        class_probs['Positive'] = class_probs['Positive'] + reduction * 0.75
                        class_probs['Negative'] = class_probs['Negative'] + reduction * 0.25
                    else:
                        class_probs['Negative'] = class_probs['Negative'] + reduction * 0.75
                        class_probs['Positive'] = class_probs['Positive'] + reduction * 0.25
                    
                    # Renormalize
                    total = sum(class_probs.values())
                    class_probs = {k: v / total for k, v in class_probs.items()}
            
            # PRIORITY 4: Special case - if surprise is detected at all, strongly favor positive
            if surprise_raw > 5.0:  # Lower threshold for surprise
                if class_probs['Positive'] < class_probs['Neutral'] or class_probs['Positive'] < class_probs['Negative']:
                    # Surprise detected but positive not winning - make it win
                    if class_probs['Positive'] < class_probs['Neutral']:
                        shift = (class_probs['Neutral'] - class_probs['Positive']) * 0.5
                        class_probs['Positive'] = class_probs['Positive'] + shift
                        class_probs['Neutral'] = class_probs['Neutral'] - shift
                    if class_probs['Positive'] < class_probs['Negative']:
                        shift = (class_probs['Negative'] - class_probs['Positive']) * 0.4
                        class_probs['Positive'] = class_probs['Positive'] + shift
                        class_probs['Negative'] = class_probs['Negative'] - shift
                    # Renormalize
                    total = sum(class_probs.values())
                    class_probs = {k: v / total for k, v in class_probs.items()}
            
            # PRIORITY 5: Special case - if sadness is detected, strongly favor negative
            if sad_raw > 12.0:  # Very low threshold for sadness
                if class_probs['Negative'] < class_probs['Neutral']:
                    # Sadness detected but negative not winning - make it win
                    shift = (class_probs['Neutral'] - class_probs['Negative']) * 0.5
                    class_probs['Negative'] = class_probs['Negative'] + shift
                    class_probs['Neutral'] = class_probs['Neutral'] - shift
                    # Renormalize
                    total = sum(class_probs.values())
                    class_probs = {k: v / total for k, v in class_probs.items()}
            
            # PRIORITY 4: Ensure negative requires some strength (but not too strict)
            # Only reduce negative if it's very weak AND positive is clearly stronger
            if max_negative_raw < 30.0 and max_positive_raw > max_negative_raw + 10.0:
                # Negative is very weak and positive is clearly stronger
                if class_probs['Negative'] > class_probs['Positive']:
                    # Negative is winning but shouldn't - shift to positive
                    shift = (class_probs['Negative'] - class_probs['Positive']) * 0.4
                    class_probs['Negative'] = class_probs['Negative'] - shift
                    class_probs['Positive'] = class_probs['Positive'] + shift
                    # Renormalize
                    total = sum(class_probs.values())
                    class_probs = {k: v / total for k, v in class_probs.items()}
            
            if debug:
                print(f"Mapped to 3 classes: {class_probs}")
            
            # Get predicted class and confidence
            predicted_class = max(class_probs, key=class_probs.get)
            confidence = float(class_probs[predicted_class])  # Convert to Python float
            
            # Convert all probabilities to Python floats (not numpy float32)
            class_probs_float = {
                k: float(v) for k, v in class_probs.items()
            }
            
            return {
                'emotion': predicted_class,
                'confidence': confidence,
                'probabilities': class_probs_float,
                'emoji': EMOJI_MAP.get(predicted_class, ''),
                'raw_probabilities': emotions if debug else None
            }
            
        except Exception as e:
            error_msg = str(e)
            if "Face could not be detected" in error_msg or "No face detected" in error_msg:
                return {
                    'emotion': 'No Face Detected',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': 'No face detected in the image'
                }
            else:
                return {
                    'emotion': 'Error',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': f'Error during prediction: {error_msg}'
                }
    
    def predict_batch(self, images):
        """
        Predict emotions for multiple images
        
        Args:
            images: List of images (numpy arrays, PIL Images, or file paths)
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for image in images:
            pred = self.predict(image)
            predictions.append(pred)
        return predictions
