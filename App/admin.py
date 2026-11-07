from django.contrib import admin

# Register your models here.
from .models import DiabetesAssessment,UserImageModel

admin.site.register(DiabetesAssessment)
admin.site.register(UserImageModel)