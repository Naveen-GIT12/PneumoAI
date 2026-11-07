"""
App/forms.py — All forms for PneumoAI
"""
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Profile, DoctorProfile, UserImageModel, ScanRecord, PatientVitals


# ─────────────────────────────────────────
# Registration
# ─────────────────────────────────────────
class PatientRegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=100, required=True)
    last_name  = forms.CharField(max_length=100, required=True)
    email      = forms.EmailField(required=True)

    class Meta:
        model  = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name  = self.cleaned_data['last_name']
        user.email      = self.cleaned_data['email']
        if commit:
            user.save()
            Profile.objects.get_or_create(user=user, defaults={'user_type': 'patient'})
        return user


class DoctorRegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=100, required=True)
    last_name  = forms.CharField(max_length=100, required=True)
    email      = forms.EmailField(required=True)
    doctor_registration_id = forms.CharField(max_length=100, required=True, label='Doctor Registration ID')
    clinic_id      = forms.CharField(max_length=100, required=True, label='Clinic ID')
    specialization = forms.ChoiceField(choices=[
        ('', '— Select —'), ('Pulmonology', 'Pulmonology'), ('Radiology', 'Radiology'),
        ('Internal Medicine', 'Internal Medicine'), ('Cardiology', 'Cardiology'),
        ('Emergency Medicine', 'Emergency Medicine'), ('General Practice', 'General Practice'),
        ('Other', 'Other'),
    ])
    hospital_name = forms.CharField(max_length=200, required=True)
    phone_number  = forms.CharField(max_length=15,  required=True)

    class Meta:
        model  = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name  = self.cleaned_data['last_name']
        user.email      = self.cleaned_data['email']
        if commit:
            user.save()
            p, _ = Profile.objects.get_or_create(user=user, defaults={'user_type': 'doctor'})
            p.user_type = 'doctor'; p.save()
            DoctorProfile.objects.update_or_create(user=user, defaults={
                'doctor_registration_id': self.cleaned_data['doctor_registration_id'],
                'clinic_id':              self.cleaned_data['clinic_id'],
                'specialization':         self.cleaned_data['specialization'],
                'hospital_name':          self.cleaned_data['hospital_name'],
                'phone_number':           self.cleaned_data['phone_number'],
            })
        return user


# ─────────────────────────────────────────
# Auth
# ─────────────────────────────────────────
class LoginForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False)


class OTPForm(forms.Form):
    otp = forms.CharField(max_length=6, min_length=6)


# ─────────────────────────────────────────
# Profile Update
# ─────────────────────────────────────────
class UpdateUserForm(forms.ModelForm):
    class Meta:
        model  = User
        fields = ['username', 'email']


class UpdateProfileForm(forms.ModelForm):
    class Meta:
        model  = Profile
        fields = ['avatar', 'bio']


# ─────────────────────────────────────────
# Legacy Image Upload
# ─────────────────────────────────────────
class UserImageForm(forms.ModelForm):
    class Meta:
        model  = UserImageModel
        fields = ['image']


# ─────────────────────────────────────────
# Scan Upload Forms
# ─────────────────────────────────────────
class PatientScanForm(forms.Form):
    image = forms.ImageField(
        label='Chest X-Ray Image',
        widget=forms.FileInput(attrs={'accept': 'image/*'})
    )


class DoctorScanForm(forms.Form):
    GENDER_CHOICES = [('', '— Select —'), ('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')]

    image = forms.ImageField(
        label='Chest X-Ray Image',
        widget=forms.FileInput(attrs={'accept': 'image/*'})
    )
    patient_id = forms.CharField(
        max_length=50, required=True, label='Patient ID',
        widget=forms.TextInput(attrs={'placeholder': 'e.g. PAT-2024-001'})
    )
    patient_name = forms.CharField(
        max_length=120, required=True, label='Patient Full Name',
        widget=forms.TextInput(attrs={'placeholder': 'Full name'})
    )
    patient_age = forms.IntegerField(
        min_value=0, max_value=120, required=True, label='Age (years)',
        widget=forms.NumberInput(attrs={'placeholder': '35'})
    )
    patient_gender = forms.ChoiceField(choices=GENDER_CHOICES, required=True, label='Gender')
    xray_date = forms.DateField(
        required=True, label='Date X-Ray Taken',
        widget=forms.DateInput(attrs={'type': 'date'})
    )
    clinical_notes = forms.CharField(
        required=False, label='Clinical Notes / Symptoms',
        widget=forms.Textarea(attrs={'rows': 3,
            'placeholder': 'Optional: symptoms, referral reason, clinical history…'})
    )


# ─────────────────────────────────────────
# Feature 3 — Second Opinion Request
# ─────────────────────────────────────────
class SecondOpinionForm(forms.Form):
    URGENCY_CHOICES = [
        ('low',    'Low — Not urgent'),
        ('normal', 'Normal'),
        ('urgent', 'Urgent — Need fast response'),
    ]
    doctor_id = forms.IntegerField(widget=forms.HiddenInput)
    message   = forms.CharField(
        required=False, label='Message to Doctor',
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Describe your concern or question for the doctor…'
        })
    )
    urgency   = forms.ChoiceField(choices=URGENCY_CHOICES, initial='normal')


# ─────────────────────────────────────────
# Bonus — Patient Vitals Form
# ─────────────────────────────────────────
class VitalsForm(forms.ModelForm):
    class Meta:
        model  = PatientVitals
        fields = ['temperature', 'spo2', 'heart_rate', 'respiratory_rate', 'blood_pressure', 'notes']
        widgets = {
            'temperature':      forms.NumberInput(attrs={'step': '0.1', 'placeholder': '37.0'}),
            'spo2':             forms.NumberInput(attrs={'step': '0.1', 'placeholder': '98.5'}),
            'heart_rate':       forms.NumberInput(attrs={'placeholder': '72'}),
            'respiratory_rate': forms.NumberInput(attrs={'placeholder': '16'}),
            'blood_pressure':   forms.TextInput(attrs={'placeholder': '120/80'}),
            'notes':            forms.Textarea(attrs={'rows': 2, 'placeholder': 'Any notes…'}),
        }
        labels = {
            'temperature':       'Body Temperature (°C)',
            'spo2':              'SpO₂ — Oxygen Saturation (%)',
            'heart_rate':        'Heart Rate (BPM)',
            'respiratory_rate':  'Respiratory Rate (breaths/min)',
            'blood_pressure':    'Blood Pressure (mmHg)',
            'notes':             'Notes',
        }