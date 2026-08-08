import os
import uuid
import threading
import numpy as np
from PIL import Image
import cv2
from django.conf import settings

# Thread-safe lazy model singleton
_model_lock = threading.Lock()
_disease_model = None

def get_disease_model():
    global _disease_model
    if _disease_model is None:
        with _model_lock:
            if _disease_model is None:
                import tensorflow as tf
                model_path = os.path.join(settings.BASE_DIR, "plant_disease_model.h5")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                _disease_model = tf.keras.models.load_model(model_path)
    return _disease_model


# Exact alphabetical class list matching dataset directory order
DISEASE_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

DISEASE_DETAILS = {
    "Tomato_Bacterial_spot": {
        "title_en": "Tomato - Bacterial Spot",
        "title_hi": "टमाटर - जीवाणु धब्बा रोग (Bacterial Spot)",
        "cause": "Xanthomonas vesicatoria bacterium",
        "symptoms": "Small, water-soaked spots on leaves that turn dark brown/black with yellow halos. Spots may appear on stems and fruits.",
        "organic_treatment": "Spray neem oil (5ml/L) or copper oxychloride (3g/L). Remove affected leaves and destroy infected plants.",
        "chemical_treatment": "Apply Streptocycline (1g/10L) combined with Copper Oxychloride 50% WP (2.5g/L) at 10-14 day intervals.",
        "prevention": "Use certified disease-free seeds. Avoid overhead irrigation to keep foliage dry. Practice 2-3 year crop rotation."
    },
    "Tomato_Early_blight": {
        "title_en": "Tomato - Early Blight",
        "title_hi": "टमाटर - अगेती झुलसा (Early Blight)",
        "cause": "Alternaria solani fungus",
        "symptoms": "Concentric rings ('target board' pattern) on older leaves. Leaves turn yellow and drop prematurely.",
        "organic_treatment": "Apply Trichoderma viride bio-fungicide (5g/L) or baking soda solution (1 tbsp/L with soap). Mulch soil to reduce splash.",
        "chemical_treatment": "Spray Mancozeb 75% WP (2.5g/L) or Chlorothalonil 75% WP (2g/L) upon early symptom appearance.",
        "prevention": "Maintain wider plant spacing for air circulation. Stake plants off ground. Rotate crops with non-solanaceous plants."
    },
    "Tomato_Late_blight": {
        "title_en": "Tomato - Late Blight",
        "title_hi": "टमाटर - पछेती झुलसा (Late Blight)",
        "cause": "Phytophthora infestans oomycete",
        "symptoms": "Large, dark brown water-soaked patches on leaf tips and margins with white cottony growth on undersides in humid conditions.",
        "organic_treatment": "Apply Bordeaux mixture (1%) or Copper Hydroxide spray. Promptly prune affected foliage in dry weather.",
        "chemical_treatment": "Spray Cymoxanil + Mancozeb (2g/L) or Metalaxyl + Mancozeb (2.5g/L) immediately when wet cold conditions prevail.",
        "prevention": "Ensure good field drainage. Avoid sprinkler watering. Destroy crop debris after harvest."
    },
    "Tomato_Leaf_Mold": {
        "title_en": "Tomato - Leaf Mold",
        "title_hi": "टमाटर - पत्ती फफूंद (Leaf Mold)",
        "cause": "Passalora fulva fungus",
        "symptoms": "Pale green or yellow spots on upper leaf surface with velvety olive-green to brown fungal growth underneath.",
        "organic_treatment": "Spray potassium bicarbonate (3g/L) or sulfur-based organic fungicides. Improve greenhouse ventilation.",
        "chemical_treatment": "Spray Copper Hydroxide (2.5g/L) or Difenoconazole 25% EC (0.5ml/L).",
        "prevention": "Keep relative humidity below 85%. Prune lower leaves to enhance airflow. Space rows adequately."
    },
    "Tomato_Septoria_leaf_spot": {
        "title_en": "Tomato - Septoria Leaf Spot",
        "title_hi": "टमाटर - सेप्टोरिया पत्ती धब्बा (Septoria Spot)",
        "cause": "Septoria lycopersici fungus",
        "symptoms": "Numerous small circular spots with dark brown margins and light gray/white centers containing tiny dark specks.",
        "organic_treatment": "Remove lower infected leaves immediately. Apply copper-based organic sprays every 7-10 days.",
        "chemical_treatment": "Spray Chlorothalonil 75% WP (2g/L) or Azoxystrobin 23% SC (1ml/L).",
        "prevention": "Apply organic mulch around plant base to prevent fungal spores from splashing up from soil."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "title_en": "Tomato - Two-Spotted Spider Mites",
        "title_hi": "टमाटर - दो-धब्बेदार मकड़ी (Spider Mites)",
        "cause": "Tetranychus urticae pest",
        "symptoms": "Yellow stippling/speckling on leaves, fine silk webbing under leaves and between stems. Leaves bronze and dry out.",
        "organic_treatment": "Spray strong water jet underneath leaves. Apply neem oil (5ml/L) or insecticidal soap spray.",
        "chemical_treatment": "Spray Abamectin 1.9% EC (0.5ml/L) or Spiromesifen 22.9% SC (1ml/L) ensuring thorough undersurface coverage.",
        "prevention": "Avoid dust accumulation on foliage. Maintain adequate soil moisture to prevent plant drought stress."
    },
    "Tomato__Target_Spot": {
        "title_en": "Tomato - Target Spot",
        "title_hi": "टमाटर - लक्षित धब्बा (Target Spot)",
        "cause": "Corynespora cassiicola fungus",
        "symptoms": "Pinpoint spots that enlarge into brown circular lesions with light brown centers and faint concentric rings.",
        "organic_treatment": "Spray copper sulfate solution or bio-agent Pseudomonas fluorescens (10g/L).",
        "chemical_treatment": "Spray Azoxystrobin + Difenoconazole (1ml/L) or Mancozeb (2.5g/L).",
        "prevention": "Remove crop residues after harvest. Ensure proper field spacing and weed control."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "title_en": "Tomato - Yellow Leaf Curl Virus (TYLCV)",
        "title_hi": "टमाटर - पीला पत्ती मोड़क वाइरस (TYLCV)",
        "cause": "Begomovirus transmitted by Whiteflies (Bemisia tabaci)",
        "symptoms": "Severe curling upward of leaf margins, leaf yellowing, stunting of overall plant, and drop of flowers/fruit.",
        "organic_treatment": "No cure for viral infection. Remove infected plants. Spray neem oil (5ml/L) to control whitefly vector.",
        "chemical_treatment": "Control whitefly vector by spraying Imidacloprid 17.8% SL (0.5ml/L) or Thiamethoxam 25% WG (0.3g/L).",
        "prevention": "Install yellow sticky traps (15-20/acre) for whiteflies. Use insect-proof net nurseries."
    },
    "Tomato__Tomato_mosaic_virus": {
        "title_en": "Tomato - Mosaic Virus (ToMV)",
        "title_hi": "टमाटर - मोज़ेक वाइरस (Mosaic Virus)",
        "cause": "Tobamovirus (mechanically transmitted via hands, tools, clothes)",
        "symptoms": "Light and dark green mottling/mosaic pattern on leaves, leaf distortion (blistering or fern-like shape), stunted growth.",
        "organic_treatment": "No cure for virus. Immediately rogue out and burn infected plants. Disinfect tools with milk or 10% trisodium phosphate.",
        "chemical_treatment": "No chemical virucide exists. Control aphid vectors using Acetamiprid 20% SP (0.2g/L).",
        "prevention": "Wash hands with soap before handling plants. Avoid tobacco product usage near tomato crop. Use resistant varieties."
    },
    "Tomato_healthy": {
        "title_en": "Tomato - Healthy Crop",
        "title_hi": "टमाटर - स्वस्थ फसल (Healthy)",
        "cause": "None (Healthy)",
        "symptoms": "Leaves are vibrant green, free from spots, discoloration, wilt, or pest damage.",
        "organic_treatment": "Maintain balanced organic fertilization with FYM/vermicompost and regular watering.",
        "chemical_treatment": "Apply recommended preventive balanced NPK fertilization.",
        "prevention": "Continue regular monitoring, crop rotation, and clean weed-free farming practices."
    }
}


def process_image_and_predict(uploaded_file):
    """
    Reads an uploaded image file, cleans & normalizes it for ML model prediction,
    and creates a separate visual contour segmentation overlay image.
    
    Returns a dict containing top prediction, confidence, top 3 classes, and details.
    """
    # Load image via PIL and ensure RGB
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # 1. PREPROCESSING FOR MODEL PREDICTION (EXACT MATCH TO TRAINING)
    # Target size: 224 x 224, Rescale: 1./255
    pred_img = pil_img.resize((224, 224))
    img_array = np.array(pred_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    # Predict
    model = get_disease_model()
    predictions = model.predict(input_tensor)[0]
    
    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index]) * 100.0
    raw_class_name = DISEASE_CLASSES[top_index] if top_index < len(DISEASE_CLASSES) else "Unknown"
    
    # Top 3 predictions
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_results = []
    for idx in top_3_indices:
        c_name = DISEASE_CLASSES[idx] if idx < len(DISEASE_CLASSES) else "Unknown"
        details = DISEASE_DETAILS.get(c_name, {})
        top_3_results.append({
            "class_raw": c_name,
            "title_en": details.get("title_en", c_name),
            "title_hi": details.get("title_hi", c_name),
            "confidence": round(float(predictions[idx]) * 100.0, 1)
        })
    
    # 2. VISUAL SEGMENTATION OVERLAY (NON-DESTRUCTIVE - FOR DISPLAY ONLY)
    cv_img = np.array(pil_img)
    cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    
    # Generate leaf HSV mask & contour highlighting for visual output
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    # Mask yellowish/brownish spots on leaf
    lower_spot = np.array([10, 40, 40])
    upper_spot = np.array([35, 255, 255])
    spot_mask = cv2.inRange(hsv, lower_spot, upper_spot)
    
    segmented_bgr = cv_img_bgr.copy()
    # Draw subtle red contour highlight around diseased spots
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(segmented_bgr, contours, -1, (0, 0, 255), 2)
    
    # Save segmented overlay image with unique filename
    unique_filename = f"segmented_{uuid.uuid4().hex[:10]}.jpg"
    seg_dir = os.path.join(settings.MEDIA_ROOT, "uploads", "segmented")
    os.makedirs(seg_dir, exist_ok=True)
    seg_path = os.path.join(seg_dir, unique_filename)
    cv2.imwrite(seg_path, segmented_bgr)
    
    rel_segmented_url = f"uploads/segmented/{unique_filename}"
    
    disease_info = DISEASE_DETAILS.get(raw_class_name, {
        "title_en": raw_class_name.replace("_", " "),
        "title_hi": raw_class_name.replace("_", " "),
        "cause": "Unknown",
        "symptoms": "Information pending.",
        "organic_treatment": "Consult local agricultural officer.",
        "chemical_treatment": "Consult local agricultural officer.",
        "prevention": "Maintain clean agricultural practices."
    })
    
    return {
        "class_raw": raw_class_name,
        "confidence": round(confidence, 1),
        "details": disease_info,
        "top_3": top_3_results,
        "segmented_url": rel_segmented_url
    }
