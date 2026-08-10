from django.shortcuts import render
from django.contrib import messages
from core.constants.indian_states import INDIAN_STATES
from core.services.mandi_service import MandiPriceService
from core.services.disease_service import process_image_and_predict
from core.services.segmentation_service import process_leaf_segmentation
from core.services.crop_service import recommend_crops, calculate_fertilizers
from core.services.weather_service import fetch_weather_for_city, WeatherServiceError
from core.models import DiseaseDetectionLog, ContactMessage, CropAdvisoryLog


# ---------------- Home Pages ----------------

def home(request):
    return render(request, "home.html")


def features(request):
    return render(request, "features.html")


def about(request):
    return render(request, "about.html")


def contact(request):
    """Contact form handling with database saving."""
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        email = request.POST.get("email", "").strip()
        phone = request.POST.get("phone", "").strip()
        subject = request.POST.get("subject", "General Inquiry").strip()
        message = request.POST.get("message", "").strip()

        if not name or not message:
            messages.error(request, "Please enter your name and message.")
        else:
            ContactMessage.objects.create(
                name=name,
                email=email,
                phone=phone,
                subject=subject,
                message=message
            )
            messages.success(request, "Thank you! Your query has been submitted successfully. Our team will contact you soon.")
            return render(request, "contact.html", {"submitted": True})

    return render(request, "contact.html")


# ---------------- Mandi Prices ----------------

def mandi_prices(request):
    """Live mandi prices from Agmarknet, filterable by state and commodity."""
    selected_state = request.GET.get("state", "Uttar Pradesh")
    selected_commodity = request.GET.get("commodity", "")
    is_search = request.GET.get("search") == "1"

    service = MandiPriceService()
    records = []
    commodities = []
    error_message = None
    info_message = None
    is_fallback = False
    fallback_notice = None

    if selected_state:
        commodities = service.fetch_commodities_for_state(selected_state)

    if is_search and selected_state:
        records, is_fallback, fallback_notice = service.fetch_prices(selected_state, selected_commodity or None)
        if is_fallback:
            info_message = fallback_notice
    elif not is_search:
        info_message = "Select a state and commodity, then click Search to view mandi prices."

    return render(
        request,
        "mandi_prices.html",
        {
            "states": INDIAN_STATES,
            "commodities": commodities,
            "selected_state": selected_state,
            "selected_commodity": selected_commodity,
            "records": records,
            "error_message": error_message,
            "info_message": info_message,
            "is_search": is_search,
            "is_fallback": is_fallback,
        },
    )


# ---------------- Plant Disease Detection ----------------

def detect_disease(request):
    """
    Plant leaf disease detection view.
    Preprocesses clean RGB images for TensorFlow model prediction,
    generates leaf contour overlay for visual feedback,
    runs optional U-Net leaf segmentation pipeline,
    and logs query to database.
    """
    context = {
        "result": None,
        "error": None,
        "confidence": None,
        "results": [],
        "segmented_image": None,
        "details": None,
        "unet_leaf_mask": None,
        "unet_black_bg": None,
        "unet_disease_mask": None
    }

    if request.method == "POST":
        if "image" not in request.FILES:
            context["error"] = "Please select a leaf image file to upload."
        else:
            uploaded_file = request.FILES["image"]
            try:
                # 1. Existing disease classification prediction
                res = process_image_and_predict(uploaded_file)
                
                context["result"] = res["details"]["title_en"]
                context["title_hi"] = res["details"]["title_hi"]
                context["confidence"] = res["confidence"]
                context["results"] = res["top_3"]
                context["segmented_image"] = res["segmented_url"]
                context["details"] = res["details"]

                # 2. Independent Supervised U-Net Leaf Segmentation Pipeline
                try:
                    seg_res = process_leaf_segmentation(uploaded_file)
                    context["unet_leaf_mask"] = seg_res.get("leaf_mask_url")
                    context["unet_black_bg"] = seg_res.get("black_bg_url")
                    context["unet_disease_mask"] = seg_res.get("disease_mask_url")
                except Exception:
                    pass

                # Log detection to database
                DiseaseDetectionLog.objects.create(
                    image=uploaded_file,
                    segmented_image=res["segmented_url"],
                    predicted_disease=res["details"]["title_en"],
                    confidence=res["confidence"]
                )

            except Exception as e:
                context["error"] = f"Error analyzing leaf image: {str(e)}"

    return render(request, "upload.html", context)


# ---------------- Crop & Fertilizer Advisory ----------------

def crop_recommendation(request):
    """Soil parameter analysis & fertilizer dosage calculation view."""
    context = {
        "recommendations": None,
        "fertilizer_plan": None,
        "form_data": {}
    }

    if request.method == "POST":
        try:
            n = float(request.POST.get("n_val", 0))
            p = float(request.POST.get("p_val", 0))
            k = float(request.POST.get("k_val", 0))
            ph = float(request.POST.get("ph_val", 6.5))
            soil_type = request.POST.get("soil_type", "Loamy")
            area = float(request.POST.get("area", 1.0))
            area_unit = request.POST.get("area_unit", "Acre")
            selected_crop = request.POST.get("target_crop", "")

            # Get crop recommendations based on soil NPK & pH
            recommendations = recommend_crops(n, p, k, ph, soil_type)
            
            # Target crop for fertilizer calculation
            target = selected_crop if selected_crop else (recommendations[0]["name"] if recommendations else "Tomato")
            fertilizer_plan = calculate_fertilizers(target, area, area_unit)

            # Log query to DB
            CropAdvisoryLog.objects.create(
                nitrogen=n,
                phosphorus=p,
                potassium=k,
                ph_level=ph,
                soil_type=soil_type,
                recommended_crop=target
            )

            context["recommendations"] = recommendations
            context["fertilizer_plan"] = fertilizer_plan
            context["form_data"] = {
                "n": n, "p": p, "k": k, "ph": ph,
                "soil_type": soil_type, "area": area,
                "area_unit": area_unit, "target_crop": target
            }

        except ValueError:
            context["error"] = "Please enter valid numeric soil parameter values."

    return render(request, "crop_recommendation.html", context)


# ---------------- Weather & Farm Advisory ----------------

def weather_advisory(request):
    """Live weather forecast and agricultural weather advisory view."""
    city = request.GET.get("city", "Lucknow").strip()
    weather_data = None
    error_message = None

    try:
        weather_data = fetch_weather_for_city(city)
    except WeatherServiceError as exc:
        error_message = exc.message

    return render(
        request,
        "weather.html",
        {
            "weather": weather_data,
            "error_message": error_message,
            "selected_city": city
        }
    )


# ---------------- Government Schemes ----------------

def government_schemes(request):
    """Portal for Indian Government Farmer Welfare Schemes."""
    return render(request, "schemes.html")


# ---------------- Multilingual Voice Assistant ----------------

def voice_assistant(request):
    """Interactive Web Speech API Assistant for Indian Farmers."""
    return render(request, "voice_assistant.html")
