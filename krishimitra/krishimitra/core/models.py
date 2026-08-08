from django.db import models

class DiseaseDetectionLog(models.Model):
    image = models.ImageField(upload_to="uploads/disease_scans/")
    segmented_image = models.ImageField(upload_to="uploads/segmented/", null=True, blank=True)
    predicted_disease = models.CharField(max_length=255)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.predicted_disease} ({self.confidence:.1f}%) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class ContactMessage(models.Model):
    name = models.CharField(max_length=150)
    email = models.EmailField(blank=True, default='')
    phone = models.CharField(max_length=20, blank=True, default='')
    subject = models.CharField(max_length=255, default='General Inquiry')
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_resolved = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Message from {self.name} - {self.subject} ({self.created_at.strftime('%Y-%m-%d')})"


class CropAdvisoryLog(models.Model):
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    ph_level = models.FloatField()
    soil_type = models.CharField(max_length=100, blank=True, default='')
    recommended_crop = models.CharField(max_length=150)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.recommended_crop} - N:{self.nitrogen} P:{self.phosphorus} K:{self.potassium} pH:{self.ph_level}"
