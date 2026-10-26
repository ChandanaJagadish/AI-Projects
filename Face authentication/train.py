#What libraries I am using and why
# DeepFace  - downloads and loads the FaceNet model automatically
# numpy     - used for numerical operations on face embeddings
# json      - used to store model metadata in JSON format
# os        - handles directory and file path operations
# urllib.request → downloads a sample image to verify model loading
# datetime  → adds timestamp showing when the model was prepared

from deepface import DeepFace
import numpy as np
import json
import os
import urllib.request
from datetime import datetime

# STEP 1: Download and load the FaceNet model
print("=" * 55)
print("Face Authentication — Model Preparation")
print("=" * 55)
print()
print("Model    : FaceNet")
print("Source   : Pre-trained by Google (open-source)")
print("Library  : DeepFace")
print()
print("Loading model... (first run may download ~90MB)")
print()

# STEP 2: Download a sample face image to test the model
os.makedirs("sample_images", exist_ok=True)

sample_url  = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"
sample_path = "sample_images/test_load.jpg"

# We use a simple test — if DeepFace can build a model object, it's loaded
try:
    print(" Step 1/3 : Initialising DeepFace with FaceNet model")
    model = DeepFace.build_model("Facenet")
    print("FaceNet model loaded successfully!")
    print(f"Model type : {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run: pip install deepface tf-keras")
    exit(1)

# STEP 3: Test that the model can produce embeddings
print()
print("Step 2/3 : Testing embedding extraction...")

try:
    dummy_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    result = DeepFace.represent(
        img_path       = dummy_image,
        model_name     = "Facenet",
        enforce_detection = False
    )
    embedding_length = len(result[0]["embedding"])
    print("Embedding extraction works!")
    print("Embedding size : {embedding_length} numbers per face")
    print(f"(Each face becomes a list of {embedding_length} numbers)")
except Exception as e:
    print(f"Embedding test failed: {e}")



# STEP 4: Save model info/metadata to model_info.json
print()
print("  Step 3/3 : Saving model info...")
model_info = {
    "model_name"        : "Facenet",
    "library"           : "DeepFace",
    "embedding_size"    : 128,
    "threshold"         : 0.70,
    "detector_backend"  : "opencv",
    "prepared_on"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "description"       : (
        "FaceNet pre-trained model loaded via DeepFace. "
        "Converts face images into 128-dimensional embeddings. "
        "Cosine similarity >= 0.70 means same person."
    ),
    "status"            : "ready"
}
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

print("  ✓ model_info.json saved!")

# DONE
print()
print("=" * 55)
print("  Model preparation complete!")
print("  The FaceNet model is ready to use.")
print()
print("  Next step: Run the testing API with:")
print("  → python face_auth.py")
print("  → Open http://127.0.0.1:8000/docs")
print("=" * 55)