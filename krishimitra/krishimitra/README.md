# Krishi Yoddha – AI-Based Farmer Query Support System

Frontend-only Django site scaffolded with Bootstrap and AOS for smooth scroll animations.

## Setup

1. Create venv (Windows PowerShell):
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:
```powershell
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
```

3. Run migrations and start server:
```powershell
.\\.venv\\Scripts\\python manage.py migrate
.\\.venv\\Scripts\\python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Pages
- Home (hero, CTA)
- Features (cards with placeholders)
- About (team and benefits)
- Contact (form UI only)

## Tech
- Django templates
- Bootstrap 5, AOS
- Static files under `static/`, templates under `templates/`


