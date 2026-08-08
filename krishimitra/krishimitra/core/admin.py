from django.contrib import admin
from .models import DiseaseDetectionLog, ContactMessage, CropAdvisoryLog

@admin.register(DiseaseDetectionLog)
class DiseaseDetectionLogAdmin(admin.ModelAdmin):
    list_display = ('predicted_disease', 'confidence', 'created_at')
    list_filter = ('predicted_disease', 'created_at')
    search_fields = ('predicted_disease',)
    readonly_fields = ('created_at',)

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'phone', 'subject', 'created_at', 'is_resolved')
    list_filter = ('is_resolved', 'created_at')
    search_fields = ('name', 'email', 'phone', 'subject', 'message')
    list_editable = ('is_resolved',)

@admin.register(CropAdvisoryLog)
class CropAdvisoryLogAdmin(admin.ModelAdmin):
    list_display = ('recommended_crop', 'nitrogen', 'phosphorus', 'potassium', 'ph_level', 'soil_type', 'created_at')
    list_filter = ('recommended_crop', 'soil_type', 'created_at')
    search_fields = ('recommended_crop', 'soil_type')
