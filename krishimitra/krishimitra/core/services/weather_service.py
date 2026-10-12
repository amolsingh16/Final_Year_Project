"""
Live Weather Advisory Service using Open-Meteo API for Indian agricultural regions.
Includes a comprehensive Indian city/district coordinate directory and smart India-first
geocoding ranking so district headquarters (e.g. Gorakhpur, UP) are never confused
with namesake villages in other states.
"""

import re
import requests
from typing import Dict, Any, Optional, Tuple

# Comprehensive pre-mapped coordinates for Indian cities & agricultural district headquarters.
# Keys are normalized lowercase strings (also supports Hindi names via HINDI_CITY_ALIASES).
INDIAN_CITIES_COORDS: Dict[str, Dict[str, Any]] = {
    # --- Uttar Pradesh (All Major Cities & Districts) ---
    "lucknow": {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "district": "Lucknow"},
    "gorakhpur": {"name": "Gorakhpur", "lat": 26.7606, "lon": 83.3732, "state": "Uttar Pradesh", "district": "Gorakhpur"},
    "varanasi": {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh", "district": "Varanasi"},
    "banaras": {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh", "district": "Varanasi"},
    "kashi": {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh", "district": "Varanasi"},
    "kanpur": {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh", "district": "Kanpur Nagar"},
    "prayagraj": {"name": "Prayagraj", "lat": 25.4358, "lon": 81.8463, "state": "Uttar Pradesh", "district": "Prayagraj"},
    "allahabad": {"name": "Prayagraj", "lat": 25.4358, "lon": 81.8463, "state": "Uttar Pradesh", "district": "Prayagraj"},
    "agra": {"name": "Agra", "lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh", "district": "Agra"},
    "meerut": {"name": "Meerut", "lat": 28.9845, "lon": 77.7064, "state": "Uttar Pradesh", "district": "Meerut"},
    "bareilly": {"name": "Bareilly", "lat": 28.3670, "lon": 79.4304, "state": "Uttar Pradesh", "district": "Bareilly"},
    "aligarh": {"name": "Aligarh", "lat": 27.8974, "lon": 78.0880, "state": "Uttar Pradesh", "district": "Aligarh"},
    "moradabad": {"name": "Moradabad", "lat": 28.8386, "lon": 78.7733, "state": "Uttar Pradesh", "district": "Moradabad"},
    "saharanpur": {"name": "Saharanpur", "lat": 29.9680, "lon": 77.5552, "state": "Uttar Pradesh", "district": "Saharanpur"},
    "jhansi": {"name": "Jhansi", "lat": 25.4484, "lon": 78.5685, "state": "Uttar Pradesh", "district": "Jhansi"},
    "mathura": {"name": "Mathura", "lat": 27.4924, "lon": 77.6737, "state": "Uttar Pradesh", "district": "Mathura"},
    "ayodhya": {"name": "Ayodhya", "lat": 26.7922, "lon": 82.1998, "state": "Uttar Pradesh", "district": "Ayodhya"},
    "faizabad": {"name": "Ayodhya", "lat": 26.7730, "lon": 82.1458, "state": "Uttar Pradesh", "district": "Ayodhya"},
    "deoria": {"name": "Deoria", "lat": 26.5024, "lon": 83.7791, "state": "Uttar Pradesh", "district": "Deoria"},
    "kushinagar": {"name": "Kushinagar", "lat": 26.7403, "lon": 83.8889, "state": "Uttar Pradesh", "district": "Kushinagar"},
    "padrauna": {"name": "Padrauna", "lat": 26.9026, "lon": 83.9819, "state": "Uttar Pradesh", "district": "Kushinagar"},
    "maharajganj": {"name": "Maharajganj", "lat": 27.1446, "lon": 83.5620, "state": "Uttar Pradesh", "district": "Maharajganj"},
    "basti": {"name": "Basti", "lat": 26.7997, "lon": 82.7353, "state": "Uttar Pradesh", "district": "Basti"},
    "siddharthnagar": {"name": "Siddharthnagar", "lat": 27.2589, "lon": 83.0956, "state": "Uttar Pradesh", "district": "Siddharthnagar"},
    "naugarh": {"name": "Siddharthnagar", "lat": 27.2589, "lon": 83.0956, "state": "Uttar Pradesh", "district": "Siddharthnagar"},
    "sant kabir nagar": {"name": "Khalilabad", "lat": 26.7747, "lon": 83.0715, "state": "Uttar Pradesh", "district": "Sant Kabir Nagar"},
    "khalilabad": {"name": "Khalilabad", "lat": 26.7747, "lon": 83.0715, "state": "Uttar Pradesh", "district": "Sant Kabir Nagar"},
    "azamgarh": {"name": "Azamgarh", "lat": 26.0739, "lon": 83.1859, "state": "Uttar Pradesh", "district": "Azamgarh"},
    "mau": {"name": "Mau", "lat": 25.9417, "lon": 83.5611, "state": "Uttar Pradesh", "district": "Mau"},
    "ballia": {"name": "Ballia", "lat": 25.7604, "lon": 84.1471, "state": "Uttar Pradesh", "district": "Ballia"},
    "ghazipur": {"name": "Ghazipur", "lat": 25.5840, "lon": 83.5770, "state": "Uttar Pradesh", "district": "Ghazipur"},
    "jaunpur": {"name": "Jaunpur", "lat": 25.7464, "lon": 82.6837, "state": "Uttar Pradesh", "district": "Jaunpur"},
    "mirzapur": {"name": "Mirzapur", "lat": 25.1460, "lon": 82.5690, "state": "Uttar Pradesh", "district": "Mirzapur"},
    "sonbhadra": {"name": "Robertsganj", "lat": 24.6879, "lon": 83.0680, "state": "Uttar Pradesh", "district": "Sonbhadra"},
    "robertsganj": {"name": "Robertsganj", "lat": 24.6879, "lon": 83.0680, "state": "Uttar Pradesh", "district": "Sonbhadra"},
    "chandauli": {"name": "Chandauli", "lat": 25.2605, "lon": 83.2685, "state": "Uttar Pradesh", "district": "Chandauli"},
    "bhadohi": {"name": "Bhadohi", "lat": 25.3941, "lon": 82.5696, "state": "Uttar Pradesh", "district": "Bhadohi"},
    "pratapgarh": {"name": "Pratapgarh", "lat": 25.8970, "lon": 81.9442, "state": "Uttar Pradesh", "district": "Pratapgarh"},
    "sultanpur": {"name": "Sultanpur", "lat": 26.2648, "lon": 82.0727, "state": "Uttar Pradesh", "district": "Sultanpur"},
    "amethi": {"name": "Amethi", "lat": 26.1541, "lon": 81.8142, "state": "Uttar Pradesh", "district": "Amethi"},
    "rae bareli": {"name": "Rae Bareli", "lat": 26.2345, "lon": 81.2409, "state": "Uttar Pradesh", "district": "Rae Bareli"},
    "raebareli": {"name": "Rae Bareli", "lat": 26.2345, "lon": 81.2409, "state": "Uttar Pradesh", "district": "Rae Bareli"},
    "unnao": {"name": "Unnao", "lat": 26.5393, "lon": 80.4878, "state": "Uttar Pradesh", "district": "Unnao"},
    "barabanki": {"name": "Barabanki", "lat": 26.9268, "lon": 81.1834, "state": "Uttar Pradesh", "district": "Barabanki"},
    "gonda": {"name": "Gonda", "lat": 27.1340, "lon": 81.9619, "state": "Uttar Pradesh", "district": "Gonda"},
    "bahraich": {"name": "Bahraich", "lat": 27.5742, "lon": 81.5947, "state": "Uttar Pradesh", "district": "Bahraich"},
    "balrampur": {"name": "Balrampur", "lat": 27.4297, "lon": 82.1850, "state": "Uttar Pradesh", "district": "Balrampur"},
    "shravasti": {"name": "Shravasti", "lat": 27.5075, "lon": 82.0378, "state": "Uttar Pradesh", "district": "Shravasti"},
    "ambedkar nagar": {"name": "Akbarpur", "lat": 26.4288, "lon": 82.5356, "state": "Uttar Pradesh", "district": "Ambedkar Nagar"},
    "sitapur": {"name": "Sitapur", "lat": 27.5619, "lon": 80.6827, "state": "Uttar Pradesh", "district": "Sitapur"},
    "hardoi": {"name": "Hardoi", "lat": 27.3958, "lon": 80.1314, "state": "Uttar Pradesh", "district": "Hardoi"},
    "lakhimpur": {"name": "Lakhimpur Kheri", "lat": 27.9462, "lon": 80.7787, "state": "Uttar Pradesh", "district": "Lakhimpur Kheri"},
    "lakhimpur kheri": {"name": "Lakhimpur Kheri", "lat": 27.9462, "lon": 80.7787, "state": "Uttar Pradesh", "district": "Lakhimpur Kheri"},
    "shahjahanpur": {"name": "Shahjahanpur", "lat": 27.8815, "lon": 79.9120, "state": "Uttar Pradesh", "district": "Shahjahanpur"},
    "pilibhit": {"name": "Pilibhit", "lat": 28.6312, "lon": 79.8044, "state": "Uttar Pradesh", "district": "Pilibhit"},
    "budaun": {"name": "Budaun", "lat": 28.0362, "lon": 79.1263, "state": "Uttar Pradesh", "district": "Budaun"},
    "badaun": {"name": "Budaun", "lat": 28.0362, "lon": 79.1263, "state": "Uttar Pradesh", "district": "Budaun"},
    "rampur": {"name": "Rampur", "lat": 28.7983, "lon": 79.0220, "state": "Uttar Pradesh", "district": "Rampur"},
    "bijnor": {"name": "Bijnor", "lat": 29.3724, "lon": 78.1358, "state": "Uttar Pradesh", "district": "Bijnor"},
    "amroha": {"name": "Amroha", "lat": 28.9044, "lon": 78.4673, "state": "Uttar Pradesh", "district": "Amroha"},
    "sambhal": {"name": "Sambhal", "lat": 28.5849, "lon": 78.5713, "state": "Uttar Pradesh", "district": "Sambhal"},
    "muzaffarnagar": {"name": "Muzaffarnagar", "lat": 29.4727, "lon": 77.7085, "state": "Uttar Pradesh", "district": "Muzaffarnagar"},
    "shamli": {"name": "Shamli", "lat": 29.4496, "lon": 77.3129, "state": "Uttar Pradesh", "district": "Shamli"},
    "baghpat": {"name": "Baghpat", "lat": 28.9442, "lon": 77.2187, "state": "Uttar Pradesh", "district": "Baghpat"},
    "ghaziabad": {"name": "Ghaziabad", "lat": 28.6692, "lon": 77.4538, "state": "Uttar Pradesh", "district": "Ghaziabad"},
    "noida": {"name": "Noida", "lat": 28.5355, "lon": 77.3910, "state": "Uttar Pradesh", "district": "Gautam Buddha Nagar"},
    "greater noida": {"name": "Greater Noida", "lat": 28.4744, "lon": 77.5040, "state": "Uttar Pradesh", "district": "Gautam Buddha Nagar"},
    "bulandshahr": {"name": "Bulandshahr", "lat": 28.4070, "lon": 77.8498, "state": "Uttar Pradesh", "district": "Bulandshahr"},
    "hapur": {"name": "Hapur", "lat": 28.7306, "lon": 77.7759, "state": "Uttar Pradesh", "district": "Hapur"},
    "firozabad": {"name": "Firozabad", "lat": 27.1591, "lon": 78.3957, "state": "Uttar Pradesh", "district": "Firozabad"},
    "mainpuri": {"name": "Mainpuri", "lat": 27.2355, "lon": 79.0289, "state": "Uttar Pradesh", "district": "Mainpuri"},
    "etawah": {"name": "Etawah", "lat": 26.7829, "lon": 79.0155, "state": "Uttar Pradesh", "district": "Etawah"},
    "farrukhabad": {"name": "Farrukhabad", "lat": 27.3826, "lon": 79.5840, "state": "Uttar Pradesh", "district": "Farrukhabad"},
    "kannauj": {"name": "Kannauj", "lat": 27.0514, "lon": 79.9137, "state": "Uttar Pradesh", "district": "Kannauj"},
    "auraiya": {"name": "Auraiya", "lat": 26.4631, "lon": 79.5092, "state": "Uttar Pradesh", "district": "Auraiya"},
    "fatehpur": {"name": "Fatehpur", "lat": 25.9264, "lon": 80.8099, "state": "Uttar Pradesh", "district": "Fatehpur"},
    "banda": {"name": "Banda", "lat": 25.4763, "lon": 80.3395, "state": "Uttar Pradesh", "district": "Banda"},
    "chitrakoot": {"name": "Chitrakoot", "lat": 25.2022, "lon": 80.9037, "state": "Uttar Pradesh", "district": "Chitrakoot"},
    "hamirpur": {"name": "Hamirpur", "lat": 25.9554, "lon": 80.1481, "state": "Uttar Pradesh", "district": "Hamirpur"},
    "mahoba": {"name": "Mahoba", "lat": 25.2921, "lon": 79.8724, "state": "Uttar Pradesh", "district": "Mahoba"},
    "lalitpur": {"name": "Lalitpur", "lat": 24.6905, "lon": 78.4190, "state": "Uttar Pradesh", "district": "Lalitpur"},
    "orai": {"name": "Orai", "lat": 25.9904, "lon": 79.4536, "state": "Uttar Pradesh", "district": "Jalaun"},
    "jalaun": {"name": "Jalaun", "lat": 26.1430, "lon": 79.3338, "state": "Uttar Pradesh", "district": "Jalaun"},
    "etah": {"name": "Etah", "lat": 27.5588, "lon": 78.6626, "state": "Uttar Pradesh", "district": "Etah"},
    "kasganj": {"name": "Kasganj", "lat": 27.8085, "lon": 78.6458, "state": "Uttar Pradesh", "district": "Kasganj"},
    "hathras": {"name": "Hathras", "lat": 27.5980, "lon": 78.0525, "state": "Uttar Pradesh", "district": "Hathras"},

    # --- Bihar ---
    "patna": {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "state": "Bihar", "district": "Patna"},
    "gaya": {"name": "Gaya", "lat": 24.7914, "lon": 85.0002, "state": "Bihar", "district": "Gaya"},
    "muzaffarpur": {"name": "Muzaffarpur", "lat": 26.1209, "lon": 85.3647, "state": "Bihar", "district": "Muzaffarpur"},
    "bhagalpur": {"name": "Bhagalpur", "lat": 25.2425, "lon": 86.9842, "state": "Bihar", "district": "Bhagalpur"},
    "darbhanga": {"name": "Darbhanga", "lat": 26.1542, "lon": 85.8918, "state": "Bihar", "district": "Darbhanga"},
    "purnia": {"name": "Purnia", "lat": 25.7771, "lon": 87.4753, "state": "Bihar", "district": "Purnia"},
    "chhapra": {"name": "Chhapra", "lat": 25.7811, "lon": 84.7308, "state": "Bihar", "district": "Saran"},
    "saran": {"name": "Chhapra", "lat": 25.7811, "lon": 84.7308, "state": "Bihar", "district": "Saran"},
    "arrah": {"name": "Arrah", "lat": 25.5560, "lon": 84.6603, "state": "Bihar", "district": "Bhojpur"},
    "begusarai": {"name": "Begusarai", "lat": 25.4182, "lon": 86.1272, "state": "Bihar", "district": "Begusarai"},
    "samastipur": {"name": "Samastipur", "lat": 25.8629, "lon": 85.7811, "state": "Bihar", "district": "Samastipur"},
    "motihari": {"name": "Motihari", "lat": 26.6470, "lon": 84.9189, "state": "Bihar", "district": "East Champaran"},
    "bettiah": {"name": "Bettiah", "lat": 26.8018, "lon": 84.5028, "state": "Bihar", "district": "West Champaran"},
    "siwan": {"name": "Siwan", "lat": 26.2243, "lon": 84.3600, "state": "Bihar", "district": "Siwan"},
    "gopalganj": {"name": "Gopalganj", "lat": 26.4686, "lon": 84.4404, "state": "Bihar", "district": "Gopalganj"},
    "buxar": {"name": "Buxar", "lat": 25.5647, "lon": 83.9777, "state": "Bihar", "district": "Buxar"},
    "sasaram": {"name": "Sasaram", "lat": 24.9482, "lon": 84.0169, "state": "Bihar", "district": "Rohtas"},
    "hajipur": {"name": "Hajipur", "lat": 25.6858, "lon": 85.2147, "state": "Bihar", "district": "Vaishali"},
    "katihar": {"name": "Katihar", "lat": 25.5393, "lon": 87.5714, "state": "Bihar", "district": "Katihar"},
    "munger": {"name": "Munger", "lat": 25.3748, "lon": 86.4735, "state": "Bihar", "district": "Munger"},
    "nalanda": {"name": "Bihar Sharif", "lat": 25.1982, "lon": 85.5149, "state": "Bihar", "district": "Nalanda"},
    "bihar sharif": {"name": "Bihar Sharif", "lat": 25.1982, "lon": 85.5149, "state": "Bihar", "district": "Nalanda"},
    "madhubani": {"name": "Madhubani", "lat": 26.3537, "lon": 86.0717, "state": "Bihar", "district": "Madhubani"},
    "sitamarhi": {"name": "Sitamarhi", "lat": 26.5947, "lon": 85.4906, "state": "Bihar", "district": "Sitamarhi"},
    "saharsa": {"name": "Saharsa", "lat": 25.8780, "lon": 86.5960, "state": "Bihar", "district": "Saharsa"},

    # --- Punjab & Haryana ---
    "ludhiana": {"name": "Ludhiana", "lat": 30.9010, "lon": 75.8573, "state": "Punjab", "district": "Ludhiana"},
    "amritsar": {"name": "Amritsar", "lat": 31.6340, "lon": 74.8723, "state": "Punjab", "district": "Amritsar"},
    "jalandhar": {"name": "Jalandhar", "lat": 31.3260, "lon": 75.5762, "state": "Punjab", "district": "Jalandhar"},
    "patiala": {"name": "Patiala", "lat": 30.3398, "lon": 76.3869, "state": "Punjab", "district": "Patiala"},
    "bathinda": {"name": "Bathinda", "lat": 30.2110, "lon": 74.9455, "state": "Punjab", "district": "Bathinda"},
    "mohali": {"name": "Mohali", "lat": 30.7046, "lon": 76.7179, "state": "Punjab", "district": "SAS Nagar"},
    "sangrur": {"name": "Sangrur", "lat": 30.2458, "lon": 75.8421, "state": "Punjab", "district": "Sangrur"},
    "moga": {"name": "Moga", "lat": 30.8165, "lon": 75.1717, "state": "Punjab", "district": "Moga"},
    "hoshiarpur": {"name": "Hoshiarpur", "lat": 31.5143, "lon": 75.9115, "state": "Punjab", "district": "Hoshiarpur"},
    "gurdaspur": {"name": "Gurdaspur", "lat": 32.0419, "lon": 75.4053, "state": "Punjab", "district": "Gurdaspur"},
    "ferozepur": {"name": "Ferozepur", "lat": 30.9331, "lon": 74.6225, "state": "Punjab", "district": "Ferozepur"},
    "chandigarh": {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "state": "Chandigarh", "district": "Chandigarh"},
    "karnal": {"name": "Karnal", "lat": 29.6857, "lon": 76.9905, "state": "Haryana", "district": "Karnal"},
    "hisar": {"name": "Hisar", "lat": 29.1492, "lon": 75.7217, "state": "Haryana", "district": "Hisar"},
    "rohtak": {"name": "Rohtak", "lat": 28.8955, "lon": 76.6066, "state": "Haryana", "district": "Rohtak"},
    "panipat": {"name": "Panipat", "lat": 29.3909, "lon": 76.9635, "state": "Haryana", "district": "Panipat"},
    "ambala": {"name": "Ambala", "lat": 30.3782, "lon": 76.7767, "state": "Haryana", "district": "Ambala"},
    "kurukshetra": {"name": "Kurukshetra", "lat": 29.9695, "lon": 76.8783, "state": "Haryana", "district": "Kurukshetra"},
    "sirsa": {"name": "Sirsa", "lat": 29.5321, "lon": 75.0318, "state": "Haryana", "district": "Sirsa"},
    "sonipat": {"name": "Sonipat", "lat": 28.9931, "lon": 77.0151, "state": "Haryana", "district": "Sonipat"},
    "gurugram": {"name": "Gurugram", "lat": 28.4595, "lon": 77.0266, "state": "Haryana", "district": "Gurugram"},
    "gurgaon": {"name": "Gurugram", "lat": 28.4595, "lon": 77.0266, "state": "Haryana", "district": "Gurugram"},
    "faridabad": {"name": "Faridabad", "lat": 28.4089, "lon": 77.3178, "state": "Haryana", "district": "Faridabad"},

    # --- Madhya Pradesh & Chhattisgarh ---
    "bhopal": {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "district": "Bhopal"},
    "indore": {"name": "Indore", "lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "district": "Indore"},
    "jabalpur": {"name": "Jabalpur", "lat": 23.1815, "lon": 79.9864, "state": "Madhya Pradesh", "district": "Jabalpur"},
    "gwalior": {"name": "Gwalior", "lat": 26.2183, "lon": 78.1828, "state": "Madhya Pradesh", "district": "Gwalior"},
    "ujjain": {"name": "Ujjain", "lat": 23.1765, "lon": 75.7885, "state": "Madhya Pradesh", "district": "Ujjain"},
    "sagar": {"name": "Sagar", "lat": 23.8388, "lon": 78.7378, "state": "Madhya Pradesh", "district": "Sagar"},
    "rewa": {"name": "Rewa", "lat": 24.5362, "lon": 81.3037, "state": "Madhya Pradesh", "district": "Rewa"},
    "satna": {"name": "Satna", "lat": 24.6005, "lon": 80.8322, "state": "Madhya Pradesh", "district": "Satna"},
    "ratlam": {"name": "Ratlam", "lat": 23.3315, "lon": 75.0367, "state": "Madhya Pradesh", "district": "Ratlam"},
    "mandsaur": {"name": "Mandsaur", "lat": 24.0768, "lon": 75.0693, "state": "Madhya Pradesh", "district": "Mandsaur"},
    "raipur": {"name": "Raipur", "lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh", "district": "Raipur"},
    "bilaspur": {"name": "Bilaspur", "lat": 22.0797, "lon": 82.1409, "state": "Chhattisgarh", "district": "Bilaspur"},
    "durg": {"name": "Durg", "lat": 21.1904, "lon": 81.2849, "state": "Chhattisgarh", "district": "Durg"},

    # --- Rajasthan ---
    "jaipur": {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "district": "Jaipur"},
    "jodhpur": {"name": "Jodhpur", "lat": 26.2389, "lon": 73.0243, "state": "Rajasthan", "district": "Jodhpur"},
    "kota": {"name": "Kota", "lat": 25.2138, "lon": 75.8648, "state": "Rajasthan", "district": "Kota"},
    "udaipur": {"name": "Udaipur", "lat": 24.5854, "lon": 73.7125, "state": "Rajasthan", "district": "Udaipur"},
    "ajmer": {"name": "Ajmer", "lat": 26.4499, "lon": 74.6399, "state": "Rajasthan", "district": "Ajmer"},
    "bikaner": {"name": "Bikaner", "lat": 28.0229, "lon": 73.3119, "state": "Rajasthan", "district": "Bikaner"},
    "alwar": {"name": "Alwar", "lat": 27.5530, "lon": 76.6346, "state": "Rajasthan", "district": "Alwar"},
    "bharatpur": {"name": "Bharatpur", "lat": 27.2152, "lon": 77.5030, "state": "Rajasthan", "district": "Bharatpur"},
    "sikar": {"name": "Sikar", "lat": 27.6094, "lon": 75.1398, "state": "Rajasthan", "district": "Sikar"},
    "sri ganganagar": {"name": "Sri Ganganagar", "lat": 29.9038, "lon": 73.8772, "state": "Rajasthan", "district": "Sri Ganganagar"},
    "ganganagar": {"name": "Sri Ganganagar", "lat": 29.9038, "lon": 73.8772, "state": "Rajasthan", "district": "Sri Ganganagar"},

    # --- Maharashtra & Gujarat ---
    "mumbai": {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "district": "Mumbai"},
    "pune": {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "district": "Pune"},
    "nagpur": {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "state": "Maharashtra", "district": "Nagpur"},
    "nashik": {"name": "Nashik", "lat": 19.9975, "lon": 73.7898, "state": "Maharashtra", "district": "Nashik"},
    "aurangabad": {"name": "Chhatrapati Sambhajinagar", "lat": 19.8762, "lon": 75.3433, "state": "Maharashtra", "district": "Chhatrapati Sambhajinagar"},
    "solapur": {"name": "Solapur", "lat": 17.6599, "lon": 75.9064, "state": "Maharashtra", "district": "Solapur"},
    "kolhapur": {"name": "Kolhapur", "lat": 16.7050, "lon": 74.2433, "state": "Maharashtra", "district": "Kolhapur"},
    "amravati": {"name": "Amravati", "lat": 20.9320, "lon": 77.7523, "state": "Maharashtra", "district": "Amravati"},
    "akola": {"name": "Akola", "lat": 20.7002, "lon": 77.0082, "state": "Maharashtra", "district": "Akola"},
    "ahmedabad": {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "district": "Ahmedabad"},
    "surat": {"name": "Surat", "lat": 21.1702, "lon": 72.8311, "state": "Gujarat", "district": "Surat"},
    "vadodara": {"name": "Vadodara", "lat": 22.3072, "lon": 73.1812, "state": "Gujarat", "district": "Vadodara"},
    "rajkot": {"name": "Rajkot", "lat": 22.3039, "lon": 70.8022, "state": "Gujarat", "district": "Rajkot"},
    "bhavnagar": {"name": "Bhavnagar", "lat": 21.7645, "lon": 72.1519, "state": "Gujarat", "district": "Bhavnagar"},
    "junagadh": {"name": "Junagadh", "lat": 21.5222, "lon": 70.4579, "state": "Gujarat", "district": "Junagadh"},
    "gandhinagar": {"name": "Gandhinagar", "lat": 23.2156, "lon": 72.6369, "state": "Gujarat", "district": "Gandhinagar"},

    # --- East, South & North Hubs ---
    "delhi": {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "state": "Delhi", "district": "New Delhi"},
    "new delhi": {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "state": "Delhi", "district": "New Delhi"},
    "kolkata": {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "district": "Kolkata"},
    "siliguri": {"name": "Siliguri", "lat": 26.7271, "lon": 88.3953, "state": "West Bengal", "district": "Darjeeling"},
    "bardhaman": {"name": "Bardhaman", "lat": 23.2324, "lon": 87.8615, "state": "West Bengal", "district": "Purba Bardhaman"},
    "ranchi": {"name": "Ranchi", "lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "district": "Ranchi"},
    "jamshedpur": {"name": "Jamshedpur", "lat": 22.8046, "lon": 86.2029, "state": "Jharkhand", "district": "East Singhbhum"},
    "dhanbad": {"name": "Dhanbad", "lat": 23.7957, "lon": 86.4304, "state": "Jharkhand", "district": "Dhanbad"},
    "bhubaneswar": {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "state": "Odisha", "district": "Khordha"},
    "cuttack": {"name": "Cuttack", "lat": 20.4625, "lon": 85.8830, "state": "Odisha", "district": "Cuttack"},
    "guwahati": {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "state": "Assam", "district": "Kamrup"},
    "dehradun": {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "district": "Dehradun"},
    "haridwar": {"name": "Haridwar", "lat": 29.9457, "lon": 78.1642, "state": "Uttarakhand", "district": "Haridwar"},
    "haldwani": {"name": "Haldwani", "lat": 29.2183, "lon": 79.5130, "state": "Uttarakhand", "district": "Nainital"},
    "shimla": {"name": "Shimla", "lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "district": "Shimla"},
    "srinagar": {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "district": "Srinagar"},
    "jammu": {"name": "Jammu", "lat": 32.7266, "lon": 74.8570, "state": "Jammu and Kashmir", "district": "Jammu"},
    "hyderabad": {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "state": "Telangana", "district": "Hyderabad"},
    "warangal": {"name": "Warangal", "lat": 17.9689, "lon": 79.5941, "state": "Telangana", "district": "Warangal"},
    "vijayawada": {"name": "Vijayawada", "lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh", "district": "NTR"},
    "visakhapatnam": {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "district": "Visakhapatnam"},
    "guntur": {"name": "Guntur", "lat": 16.3067, "lon": 80.4365, "state": "Andhra Pradesh", "district": "Guntur"},
    "bengaluru": {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "district": "Bengaluru Urban"},
    "bangalore": {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "district": "Bengaluru Urban"},
    "mysuru": {"name": "Mysuru", "lat": 12.2958, "lon": 76.6394, "state": "Karnataka", "district": "Mysuru"},
    "hubli": {"name": "Hubballi", "lat": 15.3647, "lon": 75.1240, "state": "Karnataka", "district": "Dharwad"},
    "chennai": {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "district": "Chennai"},
    "coimbatore": {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu", "district": "Coimbatore"},
    "madurai": {"name": "Madurai", "lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu", "district": "Madurai"},
    "salem": {"name": "Salem", "lat": 11.6643, "lon": 78.1460, "state": "Tamil Nadu", "district": "Salem"},
    "kochi": {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "state": "Kerala", "district": "Ernakulam"},
    "thiruvananthapuram": {"name": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366, "state": "Kerala", "district": "Thiruvananthapuram"},
}

# Hindi script city aliases mapped to lowercase keys in INDIAN_CITIES_COORDS
HINDI_CITY_ALIASES: Dict[str, str] = {
    "गोरखपुर": "gorakhpur",
    "लखनऊ": "lucknow",
    "वाराणसी": "varanasi",
    "बनारस": "varanasi",
    "काशी": "varanasi",
    "कानपुर": "kanpur",
    "प्रयागराज": "prayagraj",
    "इलाहाबाद": "prayagraj",
    "आगरा": "agra",
    "मेरठ": "meerut",
    "बरेली": "bareilly",
    "अलीगढ़": "aligarh",
    "मुरादाबाद": "moradabad",
    "सहारनपुर": "saharanpur",
    "झांसी": "jhansi",
    "मथुरा": "mathura",
    "अयोध्या": "ayodhya",
    "फैजाबाद": "ayodhya",
    "देवरिया": "deoria",
    "कुशीनगर": "kushinagar",
    "पडरौना": "padrauna",
    "महराजगंज": "maharajganj",
    "बस्ती": "basti",
    "सिद्धार्थनगर": "siddharthnagar",
    "खलीलाबाद": "khalilabad",
    "आजमगढ़": "azamgarh",
    "मऊ": "mau",
    "बलिया": "ballia",
    "गाजीपुर": "ghazipur",
    "जौनपुर": "jaunpur",
    "मिर्जापुर": "mirzapur",
    "सोनभद्र": "sonbhadra",
    "चंदौली": "chandauli",
    "भदोही": "bhadohi",
    "प्रतापगढ़": "pratapgarh",
    "सुल्तानपुर": "sultanpur",
    "अमेठी": "amethi",
    "रायबरेली": "rae bareli",
    "उन्नाव": "unnao",
    "बाराबंकी": "barabanki",
    "गोंडा": "gonda",
    "बहराइच": "bahraich",
    "बलरामपुर": "balrampur",
    "श्रावस्ती": "shravasti",
    "सीतापुर": "sitapur",
    "हरदोई": "hardoi",
    "लखीमपुर": "lakhimpur",
    "शाहजहांपुर": "shahjahanpur",
    "पीलीभीत": "pilibhit",
    "बदायूं": "budaun",
    "रामपुर": "rampur",
    "बिजनौर": "bijnor",
    "अमरोहा": "amroha",
    "संभल": "sambhal",
    "मुजफ्फरनगर": "muzaffarnagar",
    "शामली": "shamli",
    "बागपत": "baghpat",
    "गाजियाबाद": "ghaziabad",
    "नोएडा": "noida",
    "बुलंदशहर": "bulandshahr",
    "हापुड़": "hapur",
    "फिरोजाबाद": "firozabad",
    "मैनपुरी": "mainpuri",
    "इटावा": "etawah",
    "फर्रुखाबाद": "farrukhabad",
    "कन्नौज": "kannauj",
    "औरैया": "auraiya",
    "फतेहपुर": "fatehpur",
    "बांदा": "banda",
    "चित्रकूट": "chitrakoot",
    "हमीरपुर": "hamirpur",
    "महोबा": "mahoba",
    "ललितपुर": "lalitpur",
    "जालौन": "jalaun",
    "एटा": "etah",
    "कासगंज": "kasganj",
    "हाथरस": "hathras",
    "पटना": "patna",
    "गया": "gaya",
    "मुजफ्फरपुर": "muzaffarpur",
    "भागलपुर": "bhagalpur",
    "दरभंगा": "darbhanga",
    "पूर्णिया": "purnia",
    "छपरा": "chhapra",
    "आरा": "arrah",
    "बेगूसराय": "begusarai",
    "समस्तीपुर": "samastipur",
    "मोतिहारी": "motihari",
    "बेतिया": "bettiah",
    "सीवान": "siwan",
    "गोपालगंज": "gopalganj",
    "बक्सर": "buxar",
    "सासाराम": "sasaram",
    "हाजीपुर": "hajipur",
    "कटिहार": "katihar",
    "मुंगेर": "munger",
    "नालंदा": "nalanda",
    "मधुबनी": "madhubani",
    "सीतामढ़ी": "sitamarhi",
    "सहरसा": "saharsa",
    "लुधियाना": "ludhiana",
    "अमृतसर": "amritsar",
    "जालंधर": "jalandhar",
    "पटियाला": "patiala",
    "बठिंडा": "bathinda",
    "चंडीगढ़": "chandigarh",
    "करनाल": "karnal",
    "हिसार": "hisar",
    "रोहतक": "rohtak",
    "पानीपत": "panipat",
    "अंबाला": "ambala",
    "कुरुक्षेत्र": "kurukshetra",
    "सिरसा": "sirsa",
    "सोनीपत": "sonipat",
    "गुरुग्राम": "gurugram",
    "गुड़गांव": "gurugram",
    "फरीदाबाद": "faridabad",
    "भोपाल": "bhopal",
    "इंदौर": "indore",
    "जबलपुर": "jabalpur",
    "ग्वालियर": "gwalior",
    "उज्जैन": "ujjain",
    "सागर": "sagar",
    "रीवा": "rewa",
    "सतना": "satna",
    "मंदसौर": "mandsaur",
    "रायपुर": "raipur",
    "बिलासपुर": "bilaspur",
    "जयपुर": "jaipur",
    "जोधपुर": "jodhpur",
    "कोटा": "kota",
    "उदयपुर": "udaipur",
    "अजमेर": "ajmer",
    "बीकानेर": "bikaner",
    "अलवर": "alwar",
    "भरतपुर": "bharatpur",
    "सीकर": "sikar",
    "मुंबई": "mumbai",
    "पुणे": "pune",
    "नागपुर": "nagpur",
    "नासिक": "nashik",
    "सोलापुर": "solapur",
    "कोल्हापुर": "kolhapur",
    "अहमदाबाद": "ahmedabad",
    "सूरत": "surat",
    "वडोदरा": "vadodara",
    "राजकोट": "rajkot",
    "दिल्ली": "delhi",
    "नई दिल्ली": "new delhi",
    "कोलकाता": "kolkata",
    "रांची": "ranchi",
    "भुवनेश्वर": "bhubaneswar",
    "गुवाहाटी": "guwahati",
    "देहरादून": "dehradun",
    "हरिद्वार": "haridwar",
    "शिमला": "shimla",
    "श्रीनगर": "srinagar",
    "जम्मू": "jammu",
    "हैदराबाद": "hyderabad",
    "बेंगलुरु": "bengaluru",
    "चेन्नई": "chennai",
    "कोयंबटूर": "coimbatore",
    "कोच्चि": "kochi",
}


class WeatherServiceError(Exception):
    """Raised when weather API cannot be reached or returns invalid data."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _parse_city_and_state(raw_query: str) -> Tuple[str, Optional[str]]:
    """
    Splits queries like 'Gorakhpur, Uttar Pradesh' or 'Gorakhpur UP' into
    (city_query, state_hint).
    """
    cleaned = raw_query.strip()
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        return parts[0], parts[1] if len(parts) > 1 and parts[1] else None

    # Check common state suffixes like 'Gorakhpur UP' or 'Gorakhpur Haryana'
    state_aliases = {
        "up": "Uttar Pradesh",
        "uttar pradesh": "Uttar Pradesh",
        "bihar": "Bihar",
        "mp": "Madhya Pradesh",
        "madhya pradesh": "Madhya Pradesh",
        "punjab": "Punjab",
        "haryana": "Haryana",
        "rajasthan": "Rajasthan",
        "maharashtra": "Maharashtra",
        "gujarat": "Gujarat",
        "wb": "West Bengal",
        "west bengal": "West Bengal",
        "uk": "Uttarakhand",
        "uttarakhand": "Uttarakhand",
        "hp": "Himachal Pradesh",
        "cg": "Chhattisgarh",
        "chhattisgarh": "Chhattisgarh",
        "jharkhand": "Jharkhand",
        "odisha": "Odisha",
        "karnataka": "Karnataka",
        "tamil nadu": "Tamil Nadu",
        "kerala": "Kerala",
        "telangana": "Telangana",
        "ap": "Andhra Pradesh",
        "andhra pradesh": "Andhra Pradesh",
        "assam": "Assam",
    }
    lower = cleaned.lower()
    for alias, full_state in sorted(state_aliases.items(), key=lambda x: -len(x[0])):
        if lower.endswith(" " + alias):
            city_part = cleaned[: -(len(alias) + 1)].strip()
            if city_part:
                return city_part, full_state

    return cleaned, None


def _resolve_city_coordinates(raw_query: str) -> Optional[Dict[str, Any]]:
    """
    Resolves an Indian city or district name to accurate coordinates and state.
    Prioritizes:
    1. Exact pre-mapped Indian cities/districts (unless user explicitly requested a different state).
    2. Open-Meteo Geocoding API filtered to India ('IN') with district HQ priority scoring.
    """
    city_part, state_hint = _parse_city_and_state(raw_query)
    lookup_key = HINDI_CITY_ALIASES.get(city_part.strip(), city_part.strip().lower())

    # 1. Check pre-mapped dictionary first
    if lookup_key in INDIAN_CITIES_COORDS:
        entry = INDIAN_CITIES_COORDS[lookup_key]
        if not state_hint or state_hint.lower() in entry["state"].lower():
            return entry

    # 2. Query Open-Meteo Geocoding API with count=20 and rank intelligently
    try:
        geo_res = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_part, "count": 20, "format": "json"},
            timeout=8,
        )
        geo_data = geo_res.json()
        results = geo_data.get("results") or []
        if results:
            indian_results = [r for r in results if r.get("country_code") == "IN"]
            candidates = indian_results if indian_results else results

            q_norm = re.sub(r"[^a-z0-9]", "", lookup_key)

            def score_candidate(cand: dict) -> float:
                score = 0.0
                admin1 = (cand.get("admin1") or "").lower()
                admin2 = re.sub(r"[^a-z0-9]", "", (cand.get("admin2") or "").lower())
                admin3 = re.sub(r"[^a-z0-9]", "", (cand.get("admin3") or "").lower())
                fcode = (cand.get("feature_code") or "").upper()
                pop = float(cand.get("population") or 0)

                if state_hint and state_hint.lower() in admin1:
                    score += 10000.0

                # Huge boost if District (admin2) matches the searched city name
                # e.g. 'Gorakhpur District' for 'Gorakhpur' beats a village in 'Fatehabad'
                if q_norm and (admin2 == q_norm or admin2.startswith(q_norm)):
                    score += 5000.0

                if q_norm and (admin3 == q_norm or admin3.startswith(q_norm)):
                    score += 1500.0

                # Boost administrative capitals / district headquarters
                if fcode in ("PPLC", "PPLA", "PPLA2", "PPLA3", "ADM2"):
                    score += 3000.0

                score += min(pop, 2000000.0) / 10000.0
                return score

            best = max(candidates, key=score_candidate)
            return {
                "name": best.get("name") or city_part.title(),
                "lat": best["latitude"],
                "lon": best["longitude"],
                "state": best.get("admin1") or best.get("country") or "India",
                "district": best.get("admin2") or best.get("name") or city_part.title(),
            }
    except Exception:
        pass

    # Fallback to pre-mapped dict even if state_hint was unmatched
    if lookup_key in INDIAN_CITIES_COORDS:
        return INDIAN_CITIES_COORDS[lookup_key]

    return None


def fetch_weather_for_city(city_name: str = "Lucknow") -> Dict[str, Any]:
    """
    Fetches real-time weather and 7-day forecast from Open-Meteo API.
    Raises WeatherServiceError if network request fails or city not found.
    """
    city_info = _resolve_city_coordinates(city_name or "Lucknow")

    if not city_info:
        raise WeatherServiceError(
            f"Could not locate '{city_name}'. Please enter a valid Indian city or district name (e.g. Gorakhpur, Lucknow, Patna)."
        )

    lat, lon = city_info["lat"], city_info["lon"]
    display_city = city_info.get("name", city_name.strip().title())
    display_state = city_info.get("state", "India")
    display_district = city_info.get("district", display_city)

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
            f"&hourly=relative_humidity_2m,precipitation_probability"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
            f"&timezone=Asia%2FKolkata"
        )
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        current = data.get("current_weather", {})
        daily = data.get("daily", {})

        temp = current.get("temperature", "—")
        windspeed = current.get("windspeed", "—")
        weathercode = current.get("weathercode", 0)

        # Weather code mapping
        weather_desc = _map_weather_code(weathercode)

        # Hourly humidity average
        hourly_hum = data.get("hourly", {}).get("relative_humidity_2m", [])
        humidity = round(sum(hourly_hum[:24]) / len(hourly_hum[:24])) if hourly_hum else "—"

        # Today max/min & rain probability
        max_temp = daily.get("temperature_2m_max", ["—"])[0]
        min_temp = daily.get("temperature_2m_min", ["—"])[0]
        rain_prob = daily.get("precipitation_probability_max", [0])[0]
        precip_sum = daily.get("precipitation_sum", [0])[0]

        # Agriculture Advice based on live metrics
        advisory = _generate_agri_advisory(temp, rain_prob, windspeed, humidity)

        # Build 7-day forecast array
        forecast = []
        time_list = daily.get("time", [])
        max_list = daily.get("temperature_2m_max", [])
        min_list = daily.get("temperature_2m_min", [])
        rain_list = daily.get("precipitation_probability_max", [])

        for i in range(min(7, len(time_list))):
            forecast.append({
                "date": time_list[i],
                "max_temp": max_list[i] if i < len(max_list) else "—",
                "min_temp": min_list[i] if i < len(min_list) else "—",
                "rain_prob": rain_list[i] if i < len(rain_list) else 0,
            })

        return {
            "city": display_city,
            "district": display_district,
            "state": display_state,
            "temperature": temp,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "humidity": humidity,
            "windspeed": windspeed,
            "rain_prob": rain_prob,
            "precip_sum": precip_sum,
            "weather_desc": weather_desc,
            "advisory": advisory,
            "forecast": forecast,
        }

    except Exception as exc:
        raise WeatherServiceError(
            "Unable to fetch live weather data at this moment. Please check your internet connection or try again later."
        ) from exc


def _map_weather_code(code: int) -> str:
    if code == 0:
        return "Clear Sky ☀️"
    elif code in [1, 2, 3]:
        return "Partly Cloudy ⛅"
    elif code in [45, 48]:
        return "Foggy 🌫️"
    elif code in [51, 53, 55, 61, 63, 65]:
        return "Rain / Drizzle 🌧️"
    elif code in [80, 81, 82]:
        return "Heavy Showers 🌧️"
    elif code in [95, 96, 99]:
        return "Thunderstorm 🌩️"
    return "Overcast ☁️"


def _generate_agri_advisory(temp: Any, rain_prob: int, windspeed: Any, humidity: Any) -> list:
    advisories = []

    try:
        r_prob = int(rain_prob)
    except (ValueError, TypeError):
        r_prob = 0

    try:
        w_speed = float(windspeed)
    except (ValueError, TypeError):
        w_speed = 0.0

    try:
        t_val = float(temp)
    except (ValueError, TypeError):
        t_val = 28.0

    if r_prob > 60:
        advisories.append("🌧️ **High Rain Alert**: High chance of rainfall today. Postpone pesticide spraying, fertilizer top-dressing, and irrigation.")
        advisories.append("🌾 **Drainage**: Ensure proper field drainage channels to prevent waterlogging in vegetable and pulse crop roots.")
    elif r_prob > 30:
        advisories.append("⛅ **Moderate Rain Chance**: Check local sky conditions before foliar spraying; use a sticker/spreader if spraying is urgent.")
        advisories.append("💧 **Light Irrigation**: Apply only light irrigation if soil moisture is low.")
    else:
        advisories.append("☀️ **Irrigation**: Weather is clear and suitable for scheduled field irrigation and intercultural operations.")
        advisories.append("🧪 **Spraying Window**: Favorable weather conditions for foliar nutrient, fungicide, or pesticide application during morning/evening hours.")

    if t_val >= 36:
        advisories.append("🌡️ **Heat Stress Care**: High daytime temperature — irrigate lightly in the evening and apply mulch to conserve soil moisture.")
    elif t_val <= 12:
        advisories.append("❄️ **Cold / Frost Care**: Low temperature — apply light evening irrigation to protect sensitive crops (Potato, Tomato, Mustard) from frost.")

    if w_speed > 20:
        advisories.append("💨 **High Winds**: Avoid pesticide spraying right now as wind velocity (>20 km/h) may cause spray drift.")

    return advisories
