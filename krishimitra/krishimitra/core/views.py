from django.shortcuts import render
import requests
from django.conf import settings
from urllib.parse import urlencode

import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------- Load AI Model ----------------

model = tf.keras.models.load_model("plant_disease_model.h5")

classes = [
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites",
"Tomato_Target_Spot",
"Tomato_Yellow_Leaf_Curl_Virus",
"Tomato_mosaic_virus",
"Tomato_healthy"
]


# ---------------- Home Pages ----------------

def home(request):
    return render(request, "home.html")


def features(request):
    return render(request, "features.html")


def about(request):
    return render(request, "about.html")


def contact(request):
    return render(request, "contact.html")


# ---------------- Mandi Prices ----------------

def mandi_prices(request):

    state = request.GET.get("state", "Uttar Pradesh")
    limit = request.GET.get("limit", "25")

    params = {
        "api-key": settings.DATA_GOV_API_KEY,
        "format": "json",
        "limit": limit,
    }

    query = urlencode(params)

    url = f"{settings.DATA_GOV_BASE_URL}/{settings.DATA_GOV_MANDI_RESOURCE_ID}?{query}&filters[state]={state}"

    records = []
    error_message = None

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        records = data.get("records", [])

    except Exception as e:
        error_message = str(e)

    context = {
        "state": state,
        "limit": limit,
        "records": records,
        "error_message": error_message,
    }

    return render(request, "mandi_prices.html", context)


# ---------------- Plant Disease Detection ----------------

def detect_disease(request):

    result = None
    error = None
    confidence = None
    results = []

    if request.method == "POST":

        if "image" not in request.FILES:
            error = "Please upload an image"

        else:
            try:
                img = request.FILES["image"]

                image = Image.open(img).convert("RGB")
                image = image.resize((224, 224))

                image = np.array(image) / 255.0
                image = np.expand_dims(image, axis=0)

                prediction = model.predict(image)[0]

                # Top prediction
                index = np.argmax(prediction)
                confidence = float(np.max(prediction)) * 100

                if index < len(classes):
                    result = classes[index]
                else:
                    result = "Unknown Disease"

                # Top 3 predictions
                top3 = prediction.argsort()[-3:][::-1]

                for i in top3:
                    if i < len(classes):
                        results.append({
                            "disease": classes[i],
                            "confidence": float(prediction[i]) * 100
                        })

            except Exception as e:
                error = str(e)

    context = {
        "result": result,
        "error": error,
        "confidence": confidence,
        "results": results
    }

    return render(request, "upload.html", context)