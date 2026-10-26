# Name    : Chandana Jagadish
# Task    : Task 2 - Face Authentication (Face Verification)
# Date    : April 2026
# What libraries I am using and why:
# FastAPI       → framework used to build the web API
# File          → indicates the API accepts file uploads
# UploadFile    → represents the uploaded image object
# HTTPException → used to return error responses

# uvicorn       → ASGI server used to run the FastAPI app

# numpy         → numerical operations (face embeddings are arrays)

# cv2 (OpenCV)  → image processing and face detection

# DeepFace      → extracts facial embeddings using pretrained models
#                 (Facenet model used here)

# io            → converts uploaded byte data into readable image format

# PIL (Pillow)  → image loading and conversion

# base64        → encodes image into text for JSON responses

# json          → formats structured API responses
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import cv2
from deepface import DeepFace
import io
from PIL import Image
import base64
import json
# Create the FastAPI app
app = FastAPI(
    title="Face Authentication API",
    description="Upload two face images to check if they are the same person.",
    version="1.0"
)

# FUNCTION 1: read_image
# Converts an uploaded file into a format OpenCV can work with
def read_image(upload_file: UploadFile) -> np.ndarray:
    image_bytes = upload_file.file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert("RGB")
    image_array = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return image_bgr

# FUNCTION 2: detect_face_and_box
# Detects where the face is in the image using OpenCV
def detect_face_and_box(image: np.ndarray) -> dict:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return {"found": False, "box": None}
    largest = max(faces, key=lambda f: f[2] * f[3])  
    x, y, w, h = largest

    return {
        "found": True,
        "box": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
    }



# FUNCTION 3: get_face_embedding
# Converts a face image into a list of numbers (embedding)

def get_face_embedding(image: np.ndarray) -> np.ndarray:
    embedding_result = DeepFace.represent(
        img_path=image,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="opencv"
    )
    embedding = np.array(embedding_result[0]["embedding"])

    return embedding

# FUNCTION 4: cosine_similarity
# Compares two embeddings and returns a similarity score
def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:

    dot_product = np.dot(embedding1, embedding2)

    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    similarity = dot_product / (magnitude1 * magnitude2)
    return round(float(similarity), 4)



# THE MAIN API ENDPOINT: /verify
# This is what the user calls — they upload 2 images and
# get back the verification result.
# Method: POST (because we are sending data to the server)

@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image")
):
    """
    Upload two face images to check if they belong to the same person.
    It returns:
    verification_result: "same person" or "different person"
    similarity_score: float between 0.0 and 1.0
    """

    # Step 1: Reads both uploaded images
    try:
        img1 = read_image(image1)
        img2 = read_image(image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read images: {str(e)}")

    #Step 2: Detect faces and get bounding boxes
    face1_detection = detect_face_and_box(img1)
    face2_detection = detect_face_and_box(img2)

    # Warn if no face found 
    face1_warning = None
    face2_warning = None

    if not face1_detection["found"]:
        face1_warning = "No face clearly detected in image 1 — trying anyway"

    if not face2_detection["found"]:
        face2_warning = "No face clearly detected in image 2 — trying anyway"

    # Step 3: Extract face embeddings using DeepFace + Facenet 
    try:
        embedding1 = get_face_embedding(img1)
        embedding2 = get_face_embedding(img2)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract face features: {str(e)}. Make sure images contain visible faces."
        )

    # Step 4: Calculate similarity scor
    similarity = cosine_similarity(embedding1, embedding2)

    # Step 5: Decide same person or different
    # Threshold = 0.70 means:
    # if similarity >= 0.70 - same person
    # if similarity <  0.70 - different person
    # This value is based on Facenet's recommended threshold
    THRESHOLD = 0.70

    if similarity >= THRESHOLD:
        result = "same person"
    else:
        result = "different person"

    # Step 6: Return the full result
    return {
        "verification_result" : result,
        "similarity_score"    : similarity,
        "threshold_used"      : THRESHOLD,
        "face1_bounding_box"  : face1_detection["box"],
        "face2_bounding_box"  : face2_detection["box"],
        "face1_warning"       : face1_warning,
        "face2_warning"       : face2_warning,
        "model_used"          : "FaceNet (via DeepFace)",
        "note"                : "Score closer to 1.0 = more similar faces"
    }

# Just a welcome message when you open the base URL
@app.get("/")
def home():
    return {
        "message"  : "Face Authentication API is running!",
        "usage"    : "Go to /docs to test the API in your browser",
        "endpoint" : "POST /verify — upload two face images"
    }


# RUN THE SERVER
# Once the file is runned this appears
# INFO:Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# Please add /docs to it in the web address tab of the browser
if __name__ == "__main__":
    print("=" * 55)
    print("  Face Authentication API")
    print("  Task 2 Submission")
    print("=" * 55)
    print("\n  Server starting...")
    print("  Open this in your browser:")
    print("  → http://127.0.0.1:8000/docs")
    print("\n  (Press CTRL+C to stop the server)\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )