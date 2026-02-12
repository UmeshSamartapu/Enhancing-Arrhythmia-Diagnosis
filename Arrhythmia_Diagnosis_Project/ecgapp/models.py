from django.db import models
from django.contrib.auth.models import User

class ECGRecord(models.Model):
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Prediction result
    result = models.CharField(max_length=100)
    confidence = models.FloatField(null=True, blank=True)

    # File type
    file_type = models.CharField(
        max_length=10,
        choices=[('image', 'Image'), ('csv', 'CSV')],
        default='image'
    )

    # 🫀 AI-generated heart visualization
    heart_image = models.ImageField(
        upload_to='heart_views/',
        null=True,
        blank=True
    )

    def __str__(self):
        return f"{self.filename} - {self.result}"
