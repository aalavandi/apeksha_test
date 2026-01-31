import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
from ultralytics import YOLO
# ===============================
# Configuration
# ===============================
IMAGE_SIZE = 224
EMBED_DIM = 512
NUM_CLASSES = 394  # Adjust if needed
WEIGHTS_PATH = "best_vit_magface_cattle.pth"
UPLOAD_FOLDER = "uploads1"
RESULT_FOLDER = "static1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
cattle_model = YOLO("cattle.pt")
# ===============================
# Model Definition
# ===============================
class MagFaceLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, label):
        pass  # Not needed for inference

class ViTMagFace(nn.Module):
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()
        self.fc = nn.Linear(self.backbone.embed_dim, embed_dim)
        self.magface = MagFaceLoss(embed_dim, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(self.fc(x))

# ===============================
# Load Model
# ===============================
def load_model():
    model = ViTMagFace(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

model = load_model()

def detect_cattle_in_image(image_path, conf_thresh=0.5):
    """
    Detect cattle in an image using YOLO model.
    Class IDs 19 and 20 are treated as cattle.
    """
    results = cattle_model(image_path, conf=conf_thresh)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id in [19, 20]:
                return True  # Cattle detected

    return False

# ===============================
# Preprocessing
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

# ===============================
# Cosine Similarity and Plot
# ===============================
def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2).item()

def compare_and_plot(img1_path, img2_path, output_path):
    emb1 = model(preprocess_image(img1_path))
    emb2 = model(preprocess_image(img2_path))
    similarity = cosine_similarity(emb1, emb2)
    is_same = similarity > 0.70

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")
    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    result_text = f"Similarity: {similarity:.4f} — {'✅ Same Cattle' if is_same else '❌ Different Cattle'}"
    result_color = "green" if is_same else "red"
    plt.suptitle(result_text, fontsize=16, color=result_color)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return similarity, is_same

# ===============================
# Flask App
# ===============================
app = Flask(__name__, static_folder=RESULT_FOLDER)
CORS(app)

@app.route("/cowface", methods=["POST"])
def compare():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"success": False, "error": "Both images (img1, img2) are required."}), 400

    try:
        task_id = str(uuid.uuid4())
        img1 = request.files["img1"]
        img2 = request.files["img2"]

        img1_path = os.path.join(UPLOAD_FOLDER, f"img1_{task_id}.jpg")
        img2_path = os.path.join(UPLOAD_FOLDER, f"img2_{task_id}.jpg")
        img1.save(img1_path)
        img2.save(img2_path)
        
        cattle_img1 = detect_cattle_in_image(img1_path)
        cattle_img2 = detect_cattle_in_image(img2_path)

        if not cattle_img1 or not cattle_img2:
            return jsonify({
                "success": False,
                "data": {"for": "faceid"},
                "error": {
                    "message": {
                        "message1": "No cattle detected in one or both images"
                    }
                }
            }), 400

        result_img_filename = f"result_{task_id}.png"
        result_img_path = os.path.join(RESULT_FOLDER, result_img_filename)

        similarity, is_same = compare_and_plot(img1_path, img2_path, result_img_path)
        result_img_url = f"http://0.0.0.0:5000/static/{result_img_filename}"

        return jsonify({
            "success": is_same,
            "task_id": task_id,
            "similarity_score": round(similarity, 4),
            "match": is_same,
            "message": "✅ Same Cattle" if is_same else "❌ Different Cattle",
            "result_image_url": result_img_url
        }), 200 if is_same else 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)


