"""
Intelligent Multilingual Farmer Voice Assistant Engine (Hindi, Hinglish & English).

Connects user voice/text queries to:
1. Live Weather API (Open-Meteo) for any Indian city/district
2. Live Agmarknet 2.0 Mandi Price API for any Indian state & commodity
3. ICAR Soil & Fertilizer Dosage Calculator (Urea, DAP, MOP per acre/hectare)
4. Comprehensive Crop Disease, Pest Control, Sowing Varieties & Agronomy Engine
5. Indian Government Farmer Welfare Schemes Directory
6. Live Wikipedia Knowledge API fallback for open-ended queries
"""

import re
import requests
from typing import Dict, Any, Optional, Tuple, List
from decouple import config

from core.services.weather_service import (
    fetch_weather_for_city,
    INDIAN_CITIES_COORDS,
    HINDI_CITY_ALIASES,
)
from core.services.mandi_service import MandiPriceService
from core.services.crop_service import calculate_fertilizers


# Mapping Hindi/Hinglish/English commodity keywords to canonical Agmarknet commodity names
COMMODITY_KEYWORDS: List[Tuple[List[str], str, str]] = [
    (["गेहूं", "गेहूँ", "gehu", "gehun", "wheat", "kanak"], "Wheat", "गेहूं (Wheat)"),
    (["बासमती", "basmati"], "Paddy(Basmati)", "बासमती धान (Basmati Paddy)"),
    (["धान", "चावल", "dhan", "dhaan", "chawal", "paddy", "rice"], "Paddy(Common)", "धान (Paddy)"),
    (["आलू", "aloo", "alu", "potato"], "Potato", "आलू (Potato)"),
    (["प्याज", "प्याज़", "pyaj", "pyaz", "kanda", "onion"], "Onion", "प्याज (Onion)"),
    (["टमाटर", "tamatar", "tomato"], "Tomato", "टमाटर (Tomato)"),
    (["सरसों", "लाही", "राइ", "sarson", "sarso", "mustard", "rapeseed"], "Mustard", "सरसों (Mustard)"),
    (["मक्का", "मकाई", "makka", "makai", "maize", "corn", "bhutta"], "Maize", "मक्का (Maize)"),
    (["बाजरा", "bajra", "millet", "pearl millet"], "Bajra(Pearl Millet/Cumbu)", "बाजरा (Bajra)"),
    (["ज्वार", "jowar", "sorghum"], "Jowar(Sorghum)", "ज्वार (Jowar)"),
    (["जौ", "jau", "barley"], "Barley(Jau)", "जौ (Barley)"),
    (["चना", "छोला", "chana", "gram", "chickpea"], "Bengal Gram(Gram)(Whole)", "चना (Gram)"),
    (["अरहर", "तूर", "तुअर", "arhar", "tur", "tuar", "pigeonpea", "red gram"], "Arhar(Tur/Red Gram)(Whole)", "अरहर/तूर (Arhar)"),
    (["मूंग", "मूँग", "moong", "mung", "green gram"], "Green Gram(Moong)(Whole)", "मूंग (Moong)"),
    (["उड़द", "उरद", "urad", "black gram"], "Black Gram(Urd Beans)(Whole)", "उड़द (Urad)"),
    (["मसूर", "masoor", "masur", "lentil"], "Lentil(Masur)(Whole)", "मसूर (Masoor)"),
    (["सोयाबीन", "soybean", "soyabean", "soya"], "Soyabean", "सोयाबीन (Soyabean)"),
    (["मूंगफली", "मूँगफली", "moongphali", "mungfali", "groundnut", "peanut"], "Groundnut", "मूंगफली (Groundnut)"),
    (["कपास", "नरमा", "रुई", "kapas", "narma", "cotton"], "Cotton", "कपास (Cotton)"),
    (["गन्ना", "गुड़", "चीनी", "ganna", "gur", "gud", "jaggery", "sugarcane"], "Gur(Jaggery)", "गुड़/गन्ना (Gur/Sugarcane)"),
    (["लहसुन", "lahsun", "lehsun", "garlic"], "Garlic", "लहसुन (Garlic)"),
    (["अदरक", "adrak", "ginger"], "Ginger(Green)", "अदरक (Ginger)"),
    (["मिर्च", "मिर्ची", "mirch", "mirchi", "chilli", "chili"], "Green Chilli", "हरी मिर्च (Green Chilli)"),
    (["फूलगोभी", "गोभी", "gobhi", "gobi", "cauliflower"], "Cauliflower", "फूलगोभी (Cauliflower)"),
    (["पत्तागोभी", "पत्ता गोभी", "बंदगोभी", "patta gobhi", "cabbage"], "Cabbage", "पत्तागोभी (Cabbage)"),
    (["बैंगन", "भांटा", "baingan", "brinjal", "eggplant"], "Brinjal", "बैंगन (Brinjal)"),
    (["भिंडी", "bhindi", "okra", "lady finger", "ladies finger"], "Bhindi(Ladies Finger)", "भिंडी (Bhindi)"),
    (["केला", "kela", "banana"], "Banana", "केला (Banana)"),
    (["सेब", "seb", "apple"], "Apple", "सेब (Apple)"),
]

# Mapping Hindi/Hinglish/English state keywords to INDIAN_STATES
STATE_KEYWORDS: List[Tuple[List[str], str, str]] = [
    (["उत्तर प्रदेश", "यूपी", "uttar pradesh", " u.p", " up "], "Uttar Pradesh", "उत्तर प्रदेश"),
    (["बिहार", "bihar"], "Bihar", "बिहार"),
    (["पंजाब", "punjab"], "Punjab", "पंजाब"),
    (["हरियाणा", "haryana"], "Haryana", "हरियाणा"),
    (["मध्य प्रदेश", "एमपी", "madhya pradesh", " m.p", " mp "], "Madhya Pradesh", "मध्य प्रदेश"),
    (["राजस्थान", "rajasthan"], "Rajasthan", "राजस्थान"),
    (["महाराष्ट्र", "maharashtra"], "Maharashtra", "महाराष्ट्र"),
    (["गुजरात", "gujarat"], "Gujarat", "गुजरात"),
    (["छत्तीसगढ़", "chhattisgarh", "chattisgarh"], "Chhattisgarh", "छत्तीसगढ़"),
    (["झारखंड", "jharkhand"], "Jharkhand", "झारखंड"),
    (["उत्तराखंड", "उत्तरांचल", "uttarakhand"], "Uttarakhand", "उत्तराखंड"),
    (["दिल्ली", "नई दिल्ली", "delhi"], "Delhi", "दिल्ली"),
    (["पश्चिम बंगाल", "बंगाल", "west bengal", "bengal"], "West Bengal", "पश्चिम बंगाल"),
    (["ओडिशा", "उड़ीसा", "odisha", "orissa"], "Odisha", "ओडिशा"),
    (["असम", "आसाम", "assam"], "Assam", "असम"),
    (["कर्नाटक", "karnataka"], "Karnataka", "कर्नाटक"),
    (["तमिलनाडु", "tamil nadu", "tamilnadu"], "Tamil Nadu", "तमिलनाडु"),
    (["तेलंगाना", "telangana"], "Telangana", "तेलंगाना"),
    (["आंध्र प्रदेश", "आंध्र", "andhra pradesh", "andhra"], "Andhra Pradesh", "आंध्र प्रदेश"),
    (["केरल", "kerala"], "Kerala", "केरल"),
    (["हिमाचल प्रदेश", "हिमाचल", "himachal"], "Himachal Pradesh", "हिमाचल प्रदेश"),
    (["जम्मू", "कश्मीर", "jammu", "kashmir"], "Jammu and Kashmir", "जम्मू और कश्मीर"),
    (["गोवा", "goa"], "Goa", "गोवा"),
    (["त्रिपुरा", "tripura"], "Tripura", "त्रिपुरा"),
    (["मेघालय", "meghalaya"], "Meghalaya", "मेघालय"),
    (["मणिपुर", "manipur"], "Manipur", "मणिपुर"),
    (["मिजोरम", "mizoram"], "Mizoram", "मिजोरम"),
    (["नागालैंड", "nagaland"], "Nagaland", "नागालैंड"),
    (["अरुणाचल", "arunachal"], "Arunachal Pradesh", "अरुणाचल प्रदेश"),
    (["सिक्किम", "sikkim"], "Sikkim", "सिक्किम"),
    (["चंडीगढ़", "chandigarh"], "Chandigarh", "चंडीगढ़"),
    (["पुडुचेरी", "पांडिचेरी", "puducherry", "pondicherry"], "Puducherry", "पुडुचेरी"),
    (["लद्दाख", "ladakh"], "Ladakh", "लद्दाख"),
]


# Comprehensive Crop Agronomy & Disease Knowledge Base
CROP_AGRONOMY_GUIDES: Dict[str, Dict[str, str]] = {
    "Wheat": {
        "hi": (
            "🌾 **गेहूं की उन्नत खेती एवं देखभाल**:\n"
            "• **उन्नत किस्में**: HD-2967, HD-3086, DBW-187 (करण वंदना), DBW-222, PBW-343.\n"
            "• **बुवाई एवं बीज दर**: नवंबर का महीना सर्वोत्तम है। बीज दर 40-45 किग्रा प्रति एकड़ रखें।\n"
            "• **खाद एवं उर्वरक (प्रति एकड़)**: बुवाई के समय 55 किग्रा DAP + 20 किग्रा MOP। पहली और दूसरी सिंचाई पर 45-45 किग्रा यूरिया।\n"
            "• **सिंचाई**: पहली सिंचाई (CRI अवस्था) बुवाई के 20-25 दिन बाद अवश्य करें।\n"
            "• **रोग नियंत्रण**: पीला रतुआ (Yellow Rust) दिखने पर Propiconazole 25% EC (1 ml/लीटर पानी) का छिड़काव करें।"
        ),
        "en": (
            "🌾 **Wheat Cultivation & Management**:\n"
            "• **Top Varieties**: HD-2967, HD-3086, DBW-187 (Karan Vandana), DBW-222, PBW-343.\n"
            "• **Sowing & Seed Rate**: Best sown in November using 40-45 kg seed per acre.\n"
            "• **Fertilizer per Acre**: 55 kg DAP + 20 kg MOP at sowing; 45 kg Urea each at 1st and 2nd irrigation.\n"
            "• **Critical Irrigation**: First irrigation at Crown Root Initiation (CRI) stage 20-25 days after sowing.\n"
            "• **Disease Control**: For Yellow Rust or Leaf Blight, spray Propiconazole 25% EC @ 1 ml/L."
        ),
    },
    "Paddy(Common)": {
        "hi": (
            "🌾 **धान (Paddy) की उन्नत खेती एवं सुरक्षा**:\n"
            "• **उन्नत किस्में**: पूसा बासमती-1121, 1509, स्वर्णा (MTU-7029), सहभागी, IR-64, सरजू-52.\n"
            "• **खाद एवं जिंक**: रोपाई के समय 50 किग्रा DAP + 25 किग्रा MOP + 10 किग्रा जिंक सल्फेट प्रति एकड़ डालें (खैरा रोग से बचाव हेतु)।\n"
            "• **कीट नियंत्रण**: तना छेदक (Stem Borer) और पत्ती लपेटक के लिए Cartap Hydrochloride 4G (7.5 किग्रा/एकड़) या Chlorantraniliprole 18.5% SC (60 ml/एकड़) का प्रयोग करें।\n"
            "• **रोग नियंत्रण**: झोंका रोग (Blast) या शीथ ब्लाइट के लिए Tricyclazole 75% WP (0.6 ग्राम/लीटर) या Hexaconazole (2 ml/लीटर) का छिड़काव करें।"
        ),
        "en": (
            "🌾 **Paddy (Rice) Cultivation & Protection**:\n"
            "• **Top Varieties**: Pusa Basmati-1121, 1509, Swarna (MTU-7029), IR-64, Sarju-52.\n"
            "• **Nutrients & Zinc**: Apply 50 kg DAP + 25 kg MOP + 10 kg Zinc Sulphate per acre at transplanting to prevent Khaira disease.\n"
            "• **Pest Control**: For Stem Borer & Leaf Folder, apply Cartap Hydrochloride 4G (7.5 kg/acre) or spray Chlorantraniliprole 18.5% SC (60 ml/acre).\n"
            "• **Disease Control**: For Paddy Blast or Sheath Blight, spray Tricyclazole 75% WP (0.6 g/L) or Hexaconazole (2 ml/L)."
        ),
    },
    "Tomato": {
        "hi": (
            "🍅 **टमाटर की खेती एवं रोग उपचार**:\n"
            "• **उन्नत किस्में**: अर्का रक्षक, पूसा रोहिणी, काशी विशेष, हिमसोना, नामधारी हाइब्रिड।\n"
            "• **पीले पत्ते / अगेती व पछेती झुलसा (Blight)**: निचली पुरानी पत्तियां हटाएं और Mancozeb 75% WP (2.5 ग्राम/लीटर) या Azoxystrobin + Difenoconazole (1 ml/लीटर) का छिड़काव करें।\n"
            "• **पत्ती मोड़क वायरस (Leaf Curl / मरोड़िया)**: यह सफेद मक्खी (Whitefly) से फैलता है। नियंत्रण हेतु नीम तेल (5 ml/लीटर) या Imidacloprid 17.8% SL (0.5 ml/लीटर) का छिड़काव करें और पीले चिपचिपे ट्रैप लगाएं।\n"
            "• **फल छेदक सुंडी (Fruit Borer)**: Emamectin Benzoate 5% SG (0.4 ग्राम/लीटर पानी) का स्प्रे करें।"
        ),
        "en": (
            "🍅 **Tomato Crop Care & Disease Management**:\n"
            "• **Varieties**: Arka Rakshak, Pusa Rohini, Kashi Vishesh, Himsona.\n"
            "• **Early/Late Blight & Yellowing**: Prune infected lower leaves and spray Mancozeb 75% WP (2.5 g/L) or Azoxystrobin + Difenoconazole (1 ml/L).\n"
            "• **Yellow Leaf Curl Virus**: Transmitted by Whiteflies. Install yellow sticky traps and spray Imidacloprid 17.8% SL (0.5 ml/L) or Neem Oil (5 ml/L).\n"
            "• **Fruit Borer**: Spray Emamectin Benzoate 5% SG (0.4 g/L)."
        ),
    },
    "Potato": {
        "hi": (
            "🥔 **आलू की उन्नत खेती एवं झुलसा रोग नियंत्रण**:\n"
            "• **उन्नत किस्में**: कुफरी ज्योति, कुफरी बहार (3797), कुफरी चिप्सोना, कुफरी पुखराज, कुफरी सिंदूरी।\n"
            "• **बुवाई एवं खाद**: अक्टूबर-नवंबर में बुवाई करें। प्रति एकड़ 65 किग्रा DAP, 65 किग्रा MOP और 75 किग्रा यूरिया (दो भागों में) दें।\n"
            "• **पछेती झुलसा (Late Blight) बचाव**: ठंड और कोहरे में पत्तियों पर काले-भूरे धब्बे दिखने पर तुरंत Cymoxanil + Mancozeb (2.5 ग्राम/लीटर) या Metalaxyl + Mancozeb (2.5 ग्राम/लीटर) का छिड़काव करें।"
        ),
        "en": (
            "🥔 **Potato Cultivation & Blight Protection**:\n"
            "• **Varieties**: Kufri Jyoti, Kufri Bahar (3797), Kufri Chipsona, Kufri Pukhraj.\n"
            "• **Fertilizer per Acre**: 65 kg DAP, 65 kg MOP at planting, and 75 kg Urea in split doses at earthing-up.\n"
            "• **Late Blight Treatment**: During foggy/cold weather, spray Cymoxanil + Mancozeb (2.5 g/L) or Metalaxyl + Mancozeb (2.5 g/L) every 10-12 days."
        ),
    },
    "Onion": {
        "hi": (
            "🧅 **प्याज की खेती एवं रोग प्रबंधन**:\n"
            "• **उन्नत किस्में**: भीमा सुपर, भीमा रेड, एग्रीफाउंड लाइट रेड (रबी), एग्रीफाउंड डार्क रेड (खरीफ), पूसा रेड।\n"
            "• **बैंगनी धब्बा एवं झुलसा (Purple Blotch / Stemphylium Blight)**: पत्तियों के टिप सूखने या बैंगनी धब्बे पड़ने पर Mancozeb (2.5 ग्राम/लीटर) या Tebuconazole (1 ml/लीटर) का स्प्रे करें।\n"
            "• **थ्रिप्स कीट (Thrips)**: पत्तियों में सफेद धारियां दिखने पर Fipronil 5% SC (1.5 ml/लीटर) स्टिकर के साथ छिड़कें।"
        ),
        "en": (
            "🧅 **Onion Cultivation & Protection**:\n"
            "• **Varieties**: Bhima Super, Bhima Red, Agrifound Light Red, Agrifound Dark Red, Pusa Red.\n"
            "• **Purple Blotch & Tip Burn**: Spray Mancozeb (2.5 g/L) or Tebuconazole (1 ml/L) with a surfactant/sticker.\n"
            "• **Thrips Control**: Spray Fipronil 5% SC (1.5 ml/L) or Spinosad (0.3 ml/L)."
        ),
    },
    "Mustard": {
        "hi": (
            "🌼 **सरसों (Mustard) की उन्नत खेती**:\n"
            "• **उन्नत किस्में**: पूसा बोल्ड, पूसा विजय, RH-749, गिरिराज (DRMRIJ-31), पायनियर 45S46.\n"
            "• **सल्फर का महत्व**: तेल की मात्रा और दाने की चमक बढ़ाने के लिए बुवाई के समय 10-15 किग्रा बेंटोनाइट सल्फर प्रति एकड़ अवश्य डालें (या SSP खाद का प्रयोग करें)।\n"
            "• **माहू / चेपा कीट (Aphid)**: फूल और फलियों पर माहू लगने पर Thiamethoxam 25% WG (0.4 ग्राम/लीटर) या Imidacloprid (0.5 ml/लीटर) का छिड़काव करें।"
        ),
        "en": (
            "🌼 **Mustard Cultivation & Pest Care**:\n"
            "• **Varieties**: Pusa Bold, RH-749, Giriraj (DRMRIJ-31), Pioneer 45S46.\n"
            "• **Sulphur Nutrition**: Apply 10-15 kg Bentonite Sulphur per acre (or use Single Super Phosphate) to boost oil content by 2-3%.\n"
            "• **Mustard Aphid (Mahu)**: Spray Thiamethoxam 25% WG (0.4 g/L) or Imidacloprid 17.8% SL (0.5 ml/L) during evening hours."
        ),
    },
    "Gur(Jaggery)": {
        "hi": (
            "🎋 **गन्ने (Sugarcane) की उन्नत खेती एवं लाल सड़न रोकथाम**:\n"
            "• **उन्नत किस्में**: Co-0118, Co-15023, CoLk-14201, CoS-13235 (Co-0238 में लाल सड़न का खतरा अधिक है, इसलिए नई किस्में अपनाएं)।\n"
            "• **बीज उपचार**: बुवाई से पूर्व गन्ने के टुकड़ों को Carbendazim (2 ग्राम/लीटर पानी) या Thiophanate Methyl के घोल में 15 मिनट डुबोकर उपचारित करें।\n"
            "• **चोटी बेधक एवं कंसुआ (Top Borer / Early Shoot Borer)**: Chlorantraniliprole 18.5% SC (150 ml/एकड़) को 400 लीटर पानी में मिलाकर जड़ों के पास ड्रेचिंग करें।"
        ),
        "en": (
            "🎋 **Sugarcane Cultivation & Red Rot Management**:\n"
            "• **Varieties**: Co-0118, Co-15023, CoLk-14201, CoS-13235.\n"
            "• **Sett Treatment**: Dip sugarcane setts in Carbendazim (2 g/L) or Thiophanate Methyl for 15 minutes before planting to prevent Red Rot.\n"
            "• **Top & Shoot Borer**: Drench root zone with Chlorantraniliprole 18.5% SC (150 ml/acre in 400 L water)."
        ),
    },
    "Maize": {
        "hi": (
            "🌽 **मक्का (Maize) की खेती एवं फॉल आर्मीवर्म नियंत्रण**:\n"
            "• **उन्नत किस्में**: संकर मक्का बायो-9637, DKC-9108, पायनियर 3396, पूसा एचएम-4.\n"
            "• **फॉल आर्मीवर्म (Fall Armyworm / सुंडी)**: मक्का की गोभ (पोंगा) में सुंडी लगने पर Emamectin Benzoate 5% SG (0.4 ग्राम/लीटर) या Spinetoram 11.7% SC (0.5 ml/लीटर) का सीधा पोंगे में छिड़काव करें।"
        ),
        "en": (
            "🌽 **Maize Cultivation & Fall Armyworm Control**:\n"
            "• **Varieties**: Bio-9637, DKC-9108, Pioneer 3396, Pusa HM-4.\n"
            "• **Fall Armyworm (FAW)**: Spray Emamectin Benzoate 5% SG (0.4 g/L) or Spinetoram 11.7% SC (0.5 ml/L) directly into the leaf whorl."
        ),
    },
    "Cotton": {
        "hi": (
            "☁️ **कपास (Cotton) की खेती एवं गुलाबी सुंडी नियंत्रण**:\n"
            "• **कीट प्रबंधन**: गुलाबी सुंडी (Pink Bollworm) की निगरानी के लिए प्रति एकड़ 5 फेरोमोन ट्रैप लगाएं। प्रकोप होने पर Profenofos 50% EC (2 ml/लीटर) या Emamectin Benzoate (0.5 ग्राम/लीटर) का छिड़काव करें।\n"
            "• **सफेद मक्खी एवं हरा तेला**: Flonicamid 50% WG (0.3 ग्राम/लीटर) या Diafenthiuron 50% WP (1.2 ग्राम/लीटर) का स्प्रे करें।"
        ),
        "en": (
            "☁️ **Cotton Cultivation & Pink Bollworm Control**:\n"
            "• **Pink Bollworm**: Install 5 pheromone traps per acre. Spray Profenofos 50% EC (2 ml/L) or Emamectin Benzoate (0.5 g/L).\n"
            "• **Whitefly & Jassids**: Spray Flonicamid 50% WG (0.3 g/L) or Diafenthiuron 50% WP (1.2 g/L)."
        ),
    },
    "Green Chilli": {
        "hi": (
            "🌶️ **मिर्च की खेती एवं मरोड़िया (Leaf Curl) नियंत्रण**:\n"
            "• **मरोड़िया / पत्ती मोड़क रोग**: यह थ्रिप्स, माइट्स और सफेद मक्खी से फैलता है। नियंत्रण के लिए Fipronil 5% SC (1.5 ml/लीटर) या Abamectin 1.9% EC (0.5 ml/लीटर) का छिड़काव करें और खेत में नीले व पीले स्टिकी ट्रैप लगाएं।\n"
            "• **फूल झड़ने से रोकने हेतु**: Planofix (NAA) 1 ml को 4.5 लीटर पानी में मिलाकर फूल आने पर स्प्रे करें।"
        ),
        "en": (
            "🌶️ **Chilli Cultivation & Leaf Curl Control**:\n"
            "• **Chilli Leaf Curl (Murda Disease)**: Caused by Thrips, Mites & Whiteflies. Spray Fipronil 5% SC (1.5 ml/L) alternating with Abamectin 1.9% EC (0.5 ml/L) and install blue/yellow sticky traps.\n"
            "• **Flower Drop Prevention**: Spray Planofix (NAA) @ 1 ml per 4.5 L of water at flowering stage."
        ),
    },
}


def _detect_language(query: str, lang_param: str = "hi-IN") -> str:
    """Returns 'hi' if query contains Devanagari characters or lang_param starts with 'hi', else 'en'."""
    if re.search(r"[\u0900-\u097F]", query):
        return "hi"
    if lang_param and lang_param.lower().startswith("hi"):
        # Check if query is purely English words without Hindi intent
        english_only_markers = ["what is", "how to", "tell me", "weather in", "price of", "mandi price", "fertilizer for"]
        q_low = query.lower()
        if any(m in q_low for m in english_only_markers):
            return "en"
        return "hi"
    return "en"


def _extract_city_from_query(query: str) -> Optional[str]:
    """Finds any Indian city/district name mentioned in the user's query (Hindi or English)."""
    q_low = query.lower()

    # 1. Check Hindi city names (longest match first)
    for hi_name, en_key in sorted(HINDI_CITY_ALIASES.items(), key=lambda x: -len(x[0])):
        if hi_name in query:
            return INDIAN_CITIES_COORDS[en_key]["name"]

    # 2. Check English city names in INDIAN_CITIES_COORDS (word boundary match)
    for en_key, info in sorted(INDIAN_CITIES_COORDS.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(en_key) + r"\b", q_low):
            return info["name"]

    # 3. Check regex patterns like "weather in <city>" or "<city> ka mausam"
    m1 = re.search(r"(?:weather\s+in|weather\s+of|temperature\s+in|forecast\s+for)\s+([a-zA-Z\s]{3,25})", q_low)
    if m1:
        cand = m1.group(1).strip()
        for stop in ["today", "tomorrow", "right now", "please"]:
            cand = cand.replace(stop, "").strip()
        if cand:
            return cand.title()

    m2 = re.search(r"([a-zA-Z]{3,20})\s+(?:ka\s+mausam|mein\s+barish|weather|temperature)", q_low)
    if m2:
        cand = m2.group(1).strip()
        if cand not in ("aaj", "kal", "wahan", "yahan", "kaisa", "live"):
            return cand.title()

    return None


def _extract_state_and_commodity(query: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (state_en, state_hi, commodity_en, commodity_hi) found in the query.
    Also infers state if a city/district name is mentioned!
    """
    q_padded = f" {query.lower()} "
    detected_state_en = None
    detected_state_hi = None

    for kw_list, st_en, st_hi in STATE_KEYWORDS:
        if any(kw in q_padded or kw in query for kw in kw_list):
            detected_state_en = st_en
            detected_state_hi = st_hi
            break

    # If state wasn't explicitly named, check if a city was named and infer its state!
    if not detected_state_en:
        city_found = _extract_city_from_query(query)
        if city_found and city_found.lower() in INDIAN_CITIES_COORDS:
            st_en = INDIAN_CITIES_COORDS[city_found.lower()]["state"]
            detected_state_en = st_en
            detected_state_hi = next((sh for _, se, sh in STATE_KEYWORDS if se == st_en), st_en)

    detected_cmdt_en = None
    detected_cmdt_hi = None
    for kw_list, c_en, c_hi in COMMODITY_KEYWORDS:
        if any(kw in q_padded or kw in query for kw in kw_list):
            detected_cmdt_en = c_en
            detected_cmdt_hi = c_hi
            break

    return detected_state_en, detected_state_hi, detected_cmdt_en, detected_cmdt_hi


def _fetch_wikipedia_summary(query: str, lang: str = "hi") -> Optional[str]:
    """Queries Wikipedia REST/Action API for open-ended agricultural or general questions."""
    # Strip conversational filler words to get clean search terms
    clean_q = query
    fillers = [
        "क्या है", "के बारे में बताएं", "के बारे में बताओ", "कैसे करें", "की जानकारी दें",
        "किसे कहते हैं", "बताइए", "बताओ", "कृपया", "what is", "tell me about", "how to",
        "explain", "in hindi", "in english", "?", "।",
    ]
    for f in fillers:
        clean_q = re.sub(re.escape(f), " ", clean_q, flags=re.IGNORECASE)
    clean_q = " ".join(clean_q.split()).strip()
    if len(clean_q) < 2:
        clean_q = query.strip()

    wiki_langs = ["hi", "en"] if lang == "hi" else ["en", "hi"]
    headers = {"User-Agent": "KrishiMitra-AgriAssistant/2.0 (Educational Project)"}

    for w_lang in wiki_langs:
        try:
            search_res = requests.get(
                f"https://{w_lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": clean_q,
                    "srlimit": 2,
                    "format": "json",
                },
                headers=headers,
                timeout=4,
            ).json()
            hits = search_res.get("query", {}).get("search", [])
            if not hits:
                continue

            title = hits[0]["title"]
            sum_res = requests.get(
                f"https://{w_lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                headers=headers,
                timeout=4,
            ).json()
            extract = (sum_res.get("extract") or "").strip()
            if extract and len(extract) > 40:
                # Keep first 3 sentences for clean voice synthesis
                sentences = re.split(r"(?<=[।.!?])\s+", extract)
                short_summary = " ".join(sentences[:3]).strip()
                return f"📚 **{title}**: {short_summary}"
        except Exception:
            continue

    return None


def generate_assistant_response(query: str, lang_code: str = "hi-IN") -> Dict[str, Any]:
    """
    Main entry point for Krishi Mitra Voice Assistant.
    Returns dict with:
    - answer: Formatted text response
    - speech_text: Clean plain text for Web Speech Synthesis (no markdown symbols)
    - category: Source badge label (e.g., 'Live Weather API', 'Live Mandi Price API', etc.)
    - lang: 'hi-IN' or 'en-IN'
    """
    q = (query or "").strip()
    if not q:
        return {
            "answer": "कृपया अपना प्रश्न बोलें या टाइप करें।",
            "speech_text": "कृपया अपना प्रश्न बोलें या टाइप करें।",
            "category": "Krishi Mitra AI",
            "lang": "hi-IN",
        }

    lang = _detect_language(q, lang_code)
    q_low = q.lower()

    def build_result(ans_text: str, category_label: str) -> Dict[str, Any]:
        # Strip markdown bold/bullets/emojis for natural TTS reading
        speech = re.sub(r"\*\*|\*|•|#", "", ans_text)
        speech = re.sub(r"https?://\S+", "", speech)
        speech = re.sub(r"\s+", " ", speech).strip()
        return {
            "answer": ans_text,
            "speech_text": speech,
            "category": category_label,
            "lang": "hi-IN" if lang == "hi" else "en-IN",
        }

    # -------------------------------------------------------------------------
    # 1. GREETINGS & IDENTITY
    # -------------------------------------------------------------------------
    if re.match(r"^(नमस्ते|नमस्कार|प्रणाम|राम राम|जय किसान|hello|hi|hey|namaste|namaskar)\b", q_low):
        if lang == "hi":
            return build_result(
                "🙏 **नमस्ते किसान भाई! मैं कृषि मित्र (Krishi Mitra) AI सहायक हूँ।**\n"
                "आप मुझसे किसी भी शहर का **ताज़ा मौसम**, किसी भी राज्य/फसल का **मंडी भाव**, **फसल रोग एवं कीट उपचार**, **खाद (यूरिया/डीएपी) की मात्रा**, या **सरकारी योजनाओं (PM-KISAN, फसल बीमा, KCC)** के बारे में पूछ सकते हैं।",
                "🤖 Krishi Mitra AI",
            )
        return build_result(
            "🙏 **Hello! I am Krishi Mitra, your AI Agricultural Assistant.**\n"
            "You can ask me about **live weather** for any Indian city, **live Agmarknet mandi prices** across all states, **crop disease & pest treatments**, **fertilizer dosage per acre**, or **government farmer schemes**.",
            "🤖 Krishi Mitra AI",
        )

    # -------------------------------------------------------------------------
    # 2. LIVE WEATHER API INTENT
    # -------------------------------------------------------------------------
    weather_markers = [
        "मौसम", "बारिश", "वर्षा", "तापमान", "ठंड", "गर्मी", "धूप", "कोहरा", "हवा", "ओले",
        "mausam", "barish", "baarish", "tapman", "weather", "temperature", "rain", "forecast", "humidity", "climate",
    ]
    if any(m in q_low for m in weather_markers):
        city = _extract_city_from_query(q) or "Gorakhpur"
        try:
            w = fetch_weather_for_city(city)
            adv_clean = " ".join(re.sub(r"\*\*", "", a) for a in w.get("advisory", [])[:2])
            if lang == "hi":
                ans = (
                    f"☁️ **{w['city']} ({w['state']}) का ताज़ा मौसम अपडेट**:\n"
                    f"• **वर्तमान तापमान**: {w['temperature']}°C (अधिकतम: {w['max_temp']}°C, न्यूनतम: {w['min_temp']}°C)\n"
                    f"• **मौसम की स्थिति**: {w['weather_desc']} | **नमी (Humidity)**: {w['humidity']}%\n"
                    f"• **बारिश की संभावना**: {w['rain_prob']}% | **हवा की गति**: {w['windspeed']} km/h\n"
                    f"• **कृषि सलाह**: {'आज बारिश की संभावना अधिक है, इसलिए कीटनाशक छिड़काव और सिंचाई टाल दें।' if int(w.get('rain_prob') or 0) > 50 else 'मौसम साफ है — यह सिंचाई, खाद डालने और दवा छिड़काव के लिए अनुकूल समय है।'}"
                )
            else:
                ans = (
                    f"☁️ **Live Weather in {w['city']} ({w['state']})**:\n"
                    f"• **Current Temperature**: {w['temperature']}°C (High: {w['max_temp']}°C / Low: {w['min_temp']}°C)\n"
                    f"• **Condition**: {w['weather_desc']} | **Humidity**: {w['humidity']}%\n"
                    f"• **Rain Probability**: {w['rain_prob']}% | **Wind Speed**: {w['windspeed']} km/h\n"
                    f"• **Farming Advisory**: {adv_clean}"
                )
            return build_result(ans, "☁️ Live Open-Meteo Weather API")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 3. LIVE MANDI PRICES API INTENT
    # -------------------------------------------------------------------------
    mandi_markers = [
        "भाव", "रेट", "मंडी", "दाम", "कीमत", "प्राइस", "मंड़ी",
        "bhav", "bhaav", "rate", "mandi", "price", "kimat", "daam",
    ]
    state_en, state_hi, cmdt_en, cmdt_hi = _extract_state_and_commodity(q)
    if any(m in q_low for m in mandi_markers) or (state_en and cmdt_en):
        target_state = state_en or "Uttar Pradesh"
        target_state_hi = state_hi or "उत्तर प्रदेश"
        service = MandiPriceService()
        records, _, _ = service.fetch_prices(target_state, cmdt_en, limit=5)

        if records:
            top = records[0]
            markets_sample = ", ".join(dict.fromkeys(r["market"] for r in records[:3]))
            modal_vals = [int(float(r["modal_price"])) for r in records if str(r.get("modal_price", "")).replace(".", "").isdigit()]
            avg_modal = round(sum(modal_vals) / len(modal_vals)) if modal_vals else top["modal_price"]
            min_p = min((int(float(r["min_price"])) for r in records if str(r.get("min_price", "")).replace(".", "").isdigit()), default=top["min_price"])
            max_p = max((int(float(r["max_price"])) for r in records if str(r.get("max_price", "")).replace(".", "").isdigit()), default=top["max_price"])
            c_label_hi = cmdt_hi or top["commodity"]
            c_label_en = cmdt_en or top["commodity"]

            if lang == "hi":
                ans = (
                    f"🛒 **{target_state_hi} में {c_label_hi} का ताज़ा मंडी भाव ({top['arrival_date']})**:\n"
                    f"• **औसत मॉडल भाव**: ₹{avg_modal} प्रति क्विंटल\n"
                    f"• **न्यूनतम से अधिकतम भाव**: ₹{min_p} – ₹{max_p} प्रति क्विंटल\n"
                    f"• **प्रमुख मंडियां ({top['district']} आदि)**: {markets_sample} (मॉडल रेट: ₹{top['modal_price']}/क्विंटल)\n"
                    f"• *(अन्य सभी जिलों के विस्तृत भाव देखने के लिए 'Mandi Prices' पेज पर जाएं।)*"
                )
            else:
                ans = (
                    f"🛒 **Live Mandi Price for {c_label_en} in {target_state} ({top['arrival_date']})**:\n"
                    f"• **Average Modal Price**: ₹{avg_modal} per Quintal\n"
                    f"• **Min – Max Range**: ₹{min_p} to ₹{max_p} per Quintal\n"
                    f"• **Top Reporting APMCs**: {markets_sample} (Modal: ₹{top['modal_price']}/Quintal)"
                )
            return build_result(ans, "🛒 Live Agmarknet 2.0 Mandi API")

    # -------------------------------------------------------------------------
    # 4. FERTILIZER / SOIL DOSAGE CALCULATOR INTENT
    # -------------------------------------------------------------------------
    fert_markers = [
        "खाद", "उर्वरक", "यूरिया", "डीएपी", "पोटाश", "एनपीके", "जिंक", "मिट्टी की जांच", "मिट्टी",
        "khad", "khaad", "urea", "dap", "mop", "potash", "npk", "fertilizer", "manure",
    ]
    if any(m in q_low for m in fert_markers):
        # Extract area if mentioned (e.g. "2 एकड़", "5 acre", "1 hectare")
        area_match = re.search(r"(\d+(?:\.\d+)?)\s*(एकड़|हेक्टेयर|acre|acres|hectare|ha|बीघा)", q_low)
        area_val = float(area_match.group(1)) if area_match else 1.0
        unit_raw = area_match.group(2) if area_match else "Acre"
        area_unit = "Hectare" if unit_raw in ("हेक्टेयर", "hectare", "ha") else "Acre"

        crop_for_fert = cmdt_en or "Wheat"
        if "धान" in q or "paddy" in q_low or "rice" in q_low:
            crop_for_fert = "Rice / Wheat"
        elif "आलू" in q or "potato" in q_low:
            crop_for_fert = "Potato"
        elif "टमाटर" in q or "tomato" in q_low:
            crop_for_fert = "Tomato"
        elif "मक्का" in q or "maize" in q_low:
            crop_for_fert = "Maize (Corn)"
        elif "कपास" in q or "cotton" in q_low:
            crop_for_fert = "Cotton"
        elif any(p in q_low for p in ["चना", "दाल", "अरहर", "मूंग", "pulse", "gram"]):
            crop_for_fert = "Pulses (Chickpea/Pigeonpea)"

        plan = calculate_fertilizers(crop_for_fert, area_val, area_unit)
        crop_display = cmdt_hi if (cmdt_hi and lang == "hi") else plan["crop_name"]

        if lang == "hi":
            ans = (
                f"🧪 **{plan['acres']} एकड़ ({crop_display}) के लिए वैज्ञानिक खाद एवं उर्वरक मात्रा (ICAR मानक)**:\n"
                f"• **डीएपी (DAP - 18:46:0)**: {plan['dap_kg']} किग्रा (लगभग **{plan['dap_bags']} बोरी**) — बुवाई के समय बेसल डोज में दें।\n"
                f"• **यूरिया (Urea - 46% N)**: {plan['urea_kg']} किग्रा (लगभग **{plan['urea_bags']} बोरी**) — 2 से 3 बराबर किस्तों में सिंचाई के बाद दें।\n"
                f"• **पोटाश (MOP - 60% K)**: {plan['mop_kg']} किग्रा (लगभग **{plan['mop_bags']} बोरी**) — बुवाई के समय दें।\n"
                f"• **सुझाव**: बुवाई से 15 दिन पहले प्रति एकड़ 4-5 टन सड़ी हुई गोबर की खाद या वर्मीकम्पोस्ट अवश्य मिलाएं।"
            )
        else:
            ans = (
                f"🧪 **Recommended Fertilizer Dosage for {plan['acres']} Acre(s) of {plan['crop_name']} (ICAR Standard)**:\n"
                f"• **DAP (18:46:0)**: {plan['dap_kg']} kg (~**{plan['dap_bags']} bags**) as basal dose at sowing.\n"
                f"• **Urea (46% N)**: {plan['urea_kg']} kg (~**{plan['urea_bags']} bags**) split across 2-3 top-dressings after irrigation.\n"
                f"• **MOP / Potash (60% K)**: {plan['mop_kg']} kg (~**{plan['mop_bags']} bags**) at sowing.\n"
                f"• **Tip**: Incorporate 4-5 tonnes of well-decomposed FYM/Vermicompost per acre 15 days before sowing."
            )
        return build_result(ans, "🧪 ICAR Soil & Fertilizer Engine")

    # -------------------------------------------------------------------------
    # 5. GOVERNMENT SCHEMES INTENT
    # -------------------------------------------------------------------------
    if any(k in q_low for k in ["pm kisan", "pm-kisan", "पीएम किसान", "सम्मान निधि", "6000", "किस्त", "kist", "samman nidhi"]):
        if lang == "hi":
            return build_result(
                "🏛️ **पीएम-किसान सम्मान निधि योजना (PM-KISAN)**:\n"
                "• **लाभ**: पात्र किसान परिवारों को ₹6,000 प्रति वर्ष (₹2,000 की 3 किस्तों में) सीधे बैंक खाते (DBT) में मिलते हैं।\n"
                "• **जरूरी दस्तावेज**: आधार कार्ड, जमीन की खतौनी, बैंक पासबुक और मोबाइल नंबर।\n"
                "• **आवेदन एवं ई-केवाईसी (e-KYC)**: आधिकारिक पोर्टल `pmkisan.gov.in` या नजदीकी जन सेवा केंद्र (CSC) पर जाकर पंजीकरण और e-KYC पूरा करें।",
                "🏛️ Govt Schemes Portal",
            )
        return build_result(
            "🏛️ **PM-KISAN Samman Nidhi Scheme**:\n"
            "• **Benefit**: ₹6,000 per year transferred in 3 equal installments of ₹2,000 directly to Aadhaar-linked bank accounts.\n"
            "• **Documents**: Aadhaar Card, Landholding papers (Khatauni), Bank Account.\n"
            "• **Apply / Check Status**: Visit `pmkisan.gov.in` or your nearest Common Service Centre (CSC) to complete registration and e-KYC.",
            "🏛️ Govt Schemes Portal",
        )

    if any(k in q_low for k in ["बीमा", "फसल बीमा", "मुआवजा", "pmfby", "fasal bima", "insurance"]):
        if lang == "hi":
            return build_result(
                "🛡️ **प्रधानमंत्री फसल बीमा योजना (PMFBY)**:\n"
                "• **लाभ**: बाढ़, सूखा, ओलावृष्टि, बेमौसम बारिश या कीट-रोग से फसल नष्ट होने पर पूर्ण बीमा सुरक्षा।\n"
                "• **प्रीमियम दर**: खरीफ फसलों के लिए केवल **2%**, रबी फसलों के लिए **1.5%**, और बागवानी/नकदी फसलों के लिए **5%**।\n"
                "• **नुकसान की सूचना**: आपदा के **72 घंटे** के भीतर Crop Insurance App या टोल-फ्री नंबर **14447** पर सूचना दें। पोर्टल: `pmfby.gov.in`",
                "🏛️ Govt Schemes Portal",
            )
        return build_result(
            "🛡️ **Pradhan Mantri Fasal Bima Yojana (PMFBY)**:\n"
            "• **Coverage**: Protects against drought, flood, hailstorm, unseasonal rains, and pest/disease outbreaks.\n"
            "• **Farmer Premium**: Only **2%** for Kharif crops, **1.5%** for Rabi crops, and **5%** for commercial/horticultural crops.\n"
            "• **Claim Reporting**: Intimate crop loss within **72 hours** via Helpline **14447** or `pmfby.gov.in`.",
            "🏛️ Govt Schemes Portal",
        )

    if any(k in q_low for k in ["केसीसी", "किसान क्रेडिट", "लोन", "ऋण", "कर्ज", "kcc", "kisan credit", "loan", "credit card"]):
        if lang == "hi":
            return build_result(
                "💳 **किसान क्रेडिट कार्ड (KCC) योजना**:\n"
                "• **लाभ**: खेती, पशुपालन और मछली पालन के लिए **₹3 लाख तक का लोन** बिना किसी गारंटी (Collateral-free) के मिलता है।\n"
                "• **ब्याज दर**: समय पर भुगतान करने पर सरकार की 3% छूट के बाद केवल **4% वार्षिक ब्याज** देना होता है।\n"
                "• **आवेदन**: अपने नजदीकी बैंक शाखा में आधार कार्ड, पैन कार्ड और जमीन की खतौनी के साथ आवेदन करें।",
                "🏛️ Govt Schemes Portal",
            )
        return build_result(
            "💳 **Kisan Credit Card (KCC) Scheme**:\n"
            "• **Credit Limit**: Collateral-free agricultural loan up to **₹3 Lakh** for crops, dairy, and fisheries.\n"
            "• **Interest Rate**: Effective interest rate of just **4% per annum** on timely repayment (with 3% Govt interest subvention).\n"
            "• **How to Apply**: Visit your nearest commercial, rural, or cooperative bank with Aadhaar and land records.",
            "🏛️ Govt Schemes Portal",
        )

    if any(k in q_low for k in ["सोलर", "कुसुम", "पंप", "solar", "kusum", "pump"]):
        if lang == "hi":
            return build_result(
                "☀️ **पीएम-कुसुम सोलर पंप योजना (PM-KUSUM)**:\n"
                "• **लाभ**: खेत में 2 HP से 10 HP तक का सोलर सिंचाई पंप लगवाने पर केंद्र एवं राज्य सरकार द्वारा **60% से 75% तक भारी सब्सिडी** दी जाती है।\n"
                "• **आवेदन**: अपने राज्य के कृषि विभाग पोर्टल (जैसे यूपी में `upagriculture.com`) या `pmkusum.mnre.gov.in` पर ऑनलाइन बुकिंग करें।",
                "🏛️ Govt Schemes Portal",
            )
        return build_result(
            "☀️ **PM-KUSUM Solar Pump Scheme**:\n"
            "• **Subsidy**: Provides **60% to 75% government subsidy** for installing standalone solar irrigation pumps (2 HP to 10 HP).\n"
            "• **Apply**: Register on your state agriculture portal or `pmkusum.mnre.gov.in`.",
            "🏛️ Govt Schemes Portal",
        )

    if any(k in q_low for k in ["योजना", "स्कीम", "सब्सिडी", "scheme", "subsidy", "yojana", "yojna"]):
        if lang == "hi":
            return build_result(
                "🏛️ **किसानों के लिए प्रमुख सरकारी योजनाएं**:\n"
                "1. **PM-KISAN**: ₹6,000/वर्ष की सीधी आर्थिक सहायता (`pmkisan.gov.in`)\n"
                "2. **PM फसल बीमा योजना (PMFBY)**: 1.5%–2% प्रीमियम पर फसल सुरक्षा (`pmfby.gov.in`)\n"
                "3. **किसान क्रेडिट कार्ड (KCC)**: 4% ब्याज पर ₹3 लाख तक कृषि ऋण\n"
                "4. **PM कृषि सिंचाई योजना**: ड्रिप और स्प्रिंकलर सिंचाई पर 55% से 80% सब्सिडी (`pmksy.gov.in`)\n"
                "5. **कृषि यंत्र सब्सिडी (SMAM)**: ट्रैक्टर, रोटावेटर व कृषि यंत्रों पर 40%–80% अनुदान (`agrimachinery.nic.in`)",
                "🏛️ Govt Schemes Portal",
            )
        return build_result(
            "🏛️ **Major Government Welfare Schemes for Farmers**:\n"
            "1. **PM-KISAN**: ₹6,000/year direct income support (`pmkisan.gov.in`)\n"
            "2. **PM Fasal Bima Yojana (PMFBY)**: Crop insurance at 1.5%–2% premium (`pmfby.gov.in`)\n"
            "3. **Kisan Credit Card (KCC)**: Up to ₹3 Lakh loan at 4% interest\n"
            "4. **PM Krishi Sinchayee Yojana**: 55%–80% subsidy on Drip & Sprinkler systems (`pmksy.gov.in`)\n"
            "5. **SMAM Machinery Subsidy**: 40%–80% subsidy on farm equipment (`agrimachinery.nic.in`)",
            "🏛️ Govt Schemes Portal",
        )

    # -------------------------------------------------------------------------
    # 6. SPECIFIC CROP AGRONOMY & DISEASE / PEST SYMPTOM ENGINE
    # -------------------------------------------------------------------------
    if cmdt_en and cmdt_en in CROP_AGRONOMY_GUIDES:
        guide = CROP_AGRONOMY_GUIDES[cmdt_en]
        return build_result(guide["hi"] if lang == "hi" else guide["en"], "🌾 ICAR Crop & Disease Engine")

    # Symptom-specific diagnostics (Yellowing, Blight, Curl, Wilt, Pests)
    if any(k in q_low for k in ["पीले", "पीला", "पीलापन", "yellow", "chlorosis"]):
        if lang == "hi":
            return build_result(
                "🍃 **पत्तियों के पीले पड़ने (Leaf Yellowing) के कारण एवं उपचार**:\n"
                "• **नाइट्रोजन या जिंक की कमी**: यदि पुरानी पत्तियां पीली हो रही हैं तो 1% यूरिया का छिड़काव करें; यदि नई पत्तियां पीली हैं तो **जिंक सल्फेट (5 ग्राम) + बुझा चूना (2.5 ग्राम) प्रति लीटर पानी** का स्प्रे करें।\n"
                "• **अगेती झुलसा / फफूंद संक्रमण (Early Blight)**: पत्तियों पर छल्लेदार धब्बे के साथ पीलापन हो तो **Mancozeb 75% WP (2.5 ग्राम/लीटर)** या **नीम तेल (5 ml/लीटर)** का छिड़काव करें।\n"
                "• **जलभराव**: खेत में अधिक पानी रुकने न दें। सटीक जांच के लिए 'Disease Detection' पेज पर पत्ती की फोटो अपलोड करें।",
                "🌿 Plant Protection Engine",
            )
        return build_result(
            "🍃 **Causes & Treatment for Leaf Yellowing**:\n"
            "• **Nutrient Deficiency**: Yellowing of older leaves indicates Nitrogen deficiency (foliar spray 1% Urea); yellowing of new top leaves indicates Zinc/Iron deficiency (spray Zinc Sulphate 5 g + Lime 2.5 g per liter).\n"
            "• **Early Blight / Fungal Spot**: If yellow halos surround brown spots, spray **Mancozeb 75% WP (2.5 g/L)** or **Neem Oil (5 ml/L)**.\n"
            "• **Tip**: Upload a leaf photo on the **Disease Detection** page for instant AI diagnosis.",
            "🌿 Plant Protection Engine",
        )

    if any(k in q_low for k in ["झुलसा", "ब्लाइट", "धब्बा", "धब्बे", "फफूंद", "blight", "fungus", "spot", "mildew"]):
        if lang == "hi":
            return build_result(
                "🔬 **झुलसा रोग एवं पत्ती धब्बा (Blight & Fungal Spot) नियंत्रण**:\n"
                "• **प्राथमिक उपचार**: संक्रमित निचली पत्तियों को तोड़कर खेत से बाहर नष्ट करें।\n"
                "• **रासायनिक नियंत्रण**: **Mancozeb 75% WP (2.5 ग्राम/लीटर)** या **Azoxystrobin + Tebuconazole (1 ml/लीटर पानी)** का 10 दिन के अंतराल पर छिड़काव करें।\n"
                "• **जैविक उपचार**: **Trichoderma viride (5 ग्राम/लीटर)** या कॉपर ऑक्सीक्लोराइड (3 ग्राम/लीटर) का प्रयोग करें।",
                "🌿 Plant Protection Engine",
            )
        return build_result(
            "🔬 **Blight & Fungal Leaf Spot Treatment**:\n"
            "• **Sanitation**: Remove and destroy heavily infected lower leaves.\n"
            "• **Fungicide Spray**: Spray **Mancozeb 75% WP (2.5 g/L)** or **Azoxystrobin + Tebuconazole (1 ml/L)** at 10-day intervals.\n"
            "• **Organic Option**: Apply **Trichoderma viride (5 g/L)** or Copper Oxychloride (3 g/L).",
            "🌿 Plant Protection Engine",
        )

    if any(k in q_low for k in ["मरोड़िया", "मोड़क", "सिकुड़", "कुकड़ा", "वायरस", "curl", "virus", "mosaic"]):
        if lang == "hi":
            return build_result(
                "🦠 **पत्ती मोड़क / मरोड़िया रोग (Leaf Curl & Mosaic Virus) उपचार**:\n"
                "• **कारण**: यह रोग सफेद मक्खी (Whitefly) और थ्रिप्स कीट द्वारा फैलता है।\n"
                "• **उपचार**: अत्यधिक ग्रसित पौधों को उखाड़कर नष्ट करें। कीट नियंत्रण के लिए **Imidacloprid 17.8% SL (0.5 ml/लीटर)** या **Thiamethoxam 25% WG (0.4 ग्राम/लीटर)** या **नीम तेल 1500 PPM (5 ml/लीटर)** का छिड़काव करें और प्रति एकड़ 15 पीले चिपचिपे ट्रैप (Yellow Sticky Traps) लगाएं।",
                "🌿 Plant Protection Engine",
            )
        return build_result(
            "🦠 **Leaf Curl & Mosaic Virus Management**:\n"
            "• **Cause**: Spread by Whiteflies, Thrips, and Aphids.\n"
            "• **Control**: Uproot severely stunted plants. Install 15 Yellow Sticky Traps per acre and spray **Imidacloprid 17.8% SL (0.5 ml/L)** or **Thiamethoxam 25% WG (0.4 g/L)** or **Neem Oil (5 ml/L)**.",
            "🌿 Plant Protection Engine",
        )

    if any(k in q_low for k in ["कीड़ा", "कीट", "सुंडी", "इल्ली", "माहू", "चेपा", "मक्खी", "दीमक", "तिनका", "pest", "insect", "borer", "aphid", "termite", "caterpillar"]):
        if lang == "hi":
            return build_result(
                "🐛 **फसल कीट (सुंडी, तना छेदक, माहू एवं दीमक) नियंत्रण**:\n"
                "• **फल/तना छेदक सुंडी (Borer/Caterpillar)**: **Emamectin Benzoate 5% SG (0.4 ग्राम/लीटर)** या **Chlorantraniliprole 18.5% SC (0.3 ml/लीटर)** का स्प्रे करें।\n"
                "• **रस चूसक कीट (माहू/चेपा/सफेद मक्खी)**: **Thiamethoxam 25% WG (0.4 ग्राम/लीटर)** या **नीम तेल (5 ml/लीटर)** का छिड़काव करें।\n"
                "• **दीमक (Termite)**: सिंचाई के पानी के साथ **Chlorpyriphos 20% EC (2 लीटर/एकड़)** या ब्यूवेरिया बेसियाना (जैविक) चलाएं।",
                "🌿 Plant Protection Engine",
            )
        return build_result(
            "🐛 **Integrated Pest Management (Borers, Aphids & Termites)**:\n"
            "• **Fruit/Stem Borers & Caterpillars**: Spray **Emamectin Benzoate 5% SG (0.4 g/L)** or **Chlorantraniliprole 18.5% SC (0.3 ml/L)**.\n"
            "• **Sucking Pests (Aphids/Whitefly/Jassids)**: Spray **Thiamethoxam 25% WG (0.4 g/L)** or **Neem Oil (5 ml/L)**.\n"
            "• **Termites**: Apply **Chlorpyriphos 20% EC (2 L/acre)** with irrigation water or biological *Beauveria bassiana*.",
            "🌿 Plant Protection Engine",
        )

    if any(k in q_low for k in ["उकठा", "सूख", "जड़ गलन", "गलन", "wilt", "rot", "damping"]):
        if lang == "hi":
            return build_result(
                "🌱 **उकठा रोग एवं जड़ गलन (Wilt & Root Rot) का इलाज**:\n"
                "• **लक्षण**: पौधा अचानक ऊपर से नीचे की ओर सूखने लगता है और जड़ काली/सड़ी दिखाई देती है।\n"
                "• **जैविक उपचार**: **Trichoderma viride (1 किग्रा/एकड़)** को 50 किग्रा सड़ी गोबर की खाद में मिलाकर जड़ों के पास डालें।\n"
                "• **रासायनिक ड्रेचिंग**: **Carbendazim 12% + Mancozeb 63% WP (2 ग्राम/लीटर पानी)** या **Copper Oxychloride (3 ग्राम/लीटर)** का घोल पौधों की जड़ों में डालें (Drenching)।",
                "🌿 Plant Protection Engine",
            )
        return build_result(
            "🌱 **Wilt & Root Rot Management**:\n"
            "• **Biological Control**: Apply **Trichoderma viride (1 kg/acre)** mixed with 50 kg FYM in the root zone.\n"
            "• **Chemical Drenching**: Drench the plant base with **Carbendazim 12% + Mancozeb 63% WP (2 g/L)** or **Copper Oxychloride (3 g/L)**.",
            "🌿 Plant Protection Engine",
        )

    # -------------------------------------------------------------------------
    # 7. LIVE WIKIPEDIA KNOWLEDGE BASE FALLBACK FOR ANY OPEN-ENDED QUESTION
    # -------------------------------------------------------------------------
    wiki_summary = _fetch_wikipedia_summary(q, lang=lang)
    if wiki_summary:
        if lang == "hi":
            ans = (
                f"{wiki_summary}\n\n"
                f"💡 **कृषि मित्र टिप**: यदि आप किसी विशेष फसल का **मंडी भाव**, **मौसम**, **खाद की मात्रा** या **रोग उपचार** जानना चाहते हैं, तो फसल या शहर का नाम लिखकर पूछें (जैसे: *'गोरखपुर का मौसम'*, *'बिहार में आलू का भाव'*, *'गेहूं में यूरिया की मात्रा'*)।"
            )
        else:
            ans = (
                f"{wiki_summary}\n\n"
                f"💡 **Krishi Mitra Tip**: You can also ask specific questions like *'Weather in Gorakhpur'*, *'Potato mandi price in Bihar'*, or *'Fertilizer dose for 2 acres of Wheat'*."
            )
        return build_result(ans, "🌐 Live Knowledge Base API")

    # Final helpful contextual response if even Wikipedia has no match
    if lang == "hi":
        return build_result(
            f"🌾 **कृषि मित्र सलाह ('{q}')**:\n"
            "• **फसल सुरक्षा**: यदि फसल में रोग या कीट की समस्या है, तो ऊपर **Disease Detection** पेज पर पत्ती की फोटो अपलोड करें या लक्षण बताएं (जैसे: *पीले पत्ते, झुलसा, मरोड़िया, सुंडी, माहू*)।\n"
            "• **ताज़ा मंडी भाव व मौसम**: शहर या राज्य का नाम लेकर पूछें — जैसे *'गोरखपुर का मौसम कैसा है?'* या *'पंजाब में आलू का भाव क्या है?'*\n"
            "• **निःशुल्क किसान हेल्पलाइन**: अधिक सहायता हेतु किसान कॉल सेंटर के टोल-फ्री नंबर **1800-180-1551** पर भी संपर्क कर सकते हैं।",
            "🤖 Krishi Mitra AI",
        )
    return build_result(
        f"🌾 **Krishi Mitra Guidance ('{q}')**:\n"
        "• **Crop Diagnosis**: Upload a leaf image on the **Disease Detection** page or describe symptoms (e.g., *yellow leaves, blight, leaf curl, borer, aphids*).\n"
        "• **Live Weather & Mandi Rates**: Ask with a city or state name — e.g., *'Weather in Gorakhpur'* or *'Potato price in Punjab'*.\n"
        "• **Kisan Call Center**: You can also call the toll-free helpline **1800-180-1551** for regional officer support.",
        "🤖 Krishi Mitra AI",
    )
