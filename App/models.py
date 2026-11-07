"""
App/models.py
─────────────────────────────────────────────────────────────
Contains ALL models for PneumoAI including:
  - Profile, DoctorProfile (existing)
  - UserImageModel (existing - kept for DB compatibility)
  - ScanRecord (history tracking)
  - DoctorPatientLink (feature 2 - doctor-patient assignment)
  - SecondOpinionRequest (feature 3)
  - PatientVitals (bonus feature)
  - ReportShare (bonus - secure link sharing)
─────────────────────────────────────────────────────────────
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from PIL import Image
import uuid


# ═══════════════════════════════════════════════
# USER PROFILES
# ═══════════════════════════════════════════════

class Profile(models.Model):
    USER_TYPE_CHOICES = (('patient', 'Patient'), ('doctor', 'Doctor'),)
    user      = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar    = models.ImageField(default='default.jpg', upload_to='profile_images')
    bio       = models.TextField(blank=True)
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='patient')

    def __str__(self):
        return self.user.username

    def save(self, *args, **kwargs):
        super().save()
        try:
            img = Image.open(self.avatar.path)
            if img.height > 100 or img.width > 100:
                img.thumbnail((100, 100))
                img.save(self.avatar.path)
        except Exception:
            pass


class DoctorProfile(models.Model):
    user                   = models.OneToOneField(User, on_delete=models.CASCADE)
    doctor_registration_id = models.CharField(max_length=100, unique=True)
    clinic_id              = models.CharField(max_length=100)
    specialization         = models.CharField(max_length=100)
    hospital_name          = models.CharField(max_length=200)
    phone_number           = models.CharField(max_length=15)

    def __str__(self):
        return f"Dr. {self.user.get_full_name()} — {self.doctor_registration_id}"


# ═══════════════════════════════════════════════
# UPLOAD PATH HELPER
# ═══════════════════════════════════════════════

def scan_upload_path(instance, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return f"images/{uuid.uuid4().hex}.{ext}"


# ═══════════════════════════════════════════════
# LEGACY IMAGE MODEL (kept for Database page)
# ═══════════════════════════════════════════════

class UserImageModel(models.Model):
    image   = models.ImageField(upload_to=scan_upload_path, blank=True)
    label   = models.CharField(max_length=20, default='data')
    heatmap = models.ImageField(upload_to='heatmap/', blank=True)

    def __str__(self):
        return str(self.image)


# ═══════════════════════════════════════════════
# SCAN RECORD — Full History Entry
# ═══════════════════════════════════════════════

class ScanRecord(models.Model):
    SEVERITY_CHOICES = [
        ('None',     'None'),
        ('Low',      'Low'),
        ('Moderate', 'Moderate'),
        ('Severe',   'Severe'),
    ]

    # Who + when
    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scan_records')
    created_at = models.DateTimeField(auto_now_add=True)

    # Images
    image         = models.ImageField(upload_to=scan_upload_path)
    heatmap_image = models.ImageField(upload_to='heatmap/', blank=True, null=True)

    # AI result
    predicted_class = models.CharField(max_length=30)
    predicted_label = models.CharField(max_length=60)
    confidence      = models.FloatField(default=0.0)
    affected_text   = models.TextField(blank=True)

    # Feature 5 — Severity scoring
    severity        = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default='None')
    heatmap_area    = models.FloatField(default=0.0)   # 0.0 – 1.0 fraction of lung affected

    # Feature 6 — Gemini treatment recommendations (stored as JSON text)
    treatment_recs  = models.TextField(blank=True, null=True)

    # Doctor-only patient details
    patient_id      = models.CharField(max_length=50,  blank=True, null=True)
    patient_name    = models.CharField(max_length=120, blank=True, null=True)
    patient_age     = models.PositiveIntegerField(blank=True, null=True)
    patient_gender  = models.CharField(max_length=10,  blank=True, null=True)
    xray_date       = models.DateField(blank=True, null=True)
    clinical_notes  = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} | {self.predicted_label} | {self.created_at:%Y-%m-%d}"

    @property
    def is_doctor_scan(self):
        return bool(self.patient_id or self.patient_name)

    @property
    def confidence_display(self):
        return f"{self.confidence:.1f}%"

    @property
    def severity_colour(self):
        return {
            'None': '#10b981', 'Low': '#38bdf8',
            'Moderate': '#f97316', 'Severe': '#ef4444'
        }.get(self.severity, '#94a3b8')

    def get_treatment_recs(self):
        import json
        try:
            return json.loads(self.treatment_recs) if self.treatment_recs else None
        except Exception:
            return None


# ═══════════════════════════════════════════════
# FEATURE 2 — Doctor–Patient Assignment
# ═══════════════════════════════════════════════

class DoctorPatientLink(models.Model):
    doctor      = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assigned_patients')
    patient     = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assigned_doctors')
    assigned_at = models.DateTimeField(auto_now_add=True)
    is_active   = models.BooleanField(default=True)
    notes       = models.TextField(blank=True)

    class Meta:
        unique_together = ('doctor', 'patient')

    def __str__(self):
        return f"Dr.{self.doctor.username} → {self.patient.username}"

    @property
    def patient_scan_count(self):
        return ScanRecord.objects.filter(user=self.patient).count()

    @property
    def latest_scan(self):
        return ScanRecord.objects.filter(user=self.patient).first()


# ═══════════════════════════════════════════════
# FEATURE 3 — Second Opinion Request
# ═══════════════════════════════════════════════

class SecondOpinionRequest(models.Model):
    STATUS_CHOICES = [
        ('pending',  'Pending'),
        ('accepted', 'Accepted'),
        ('reviewed', 'Reviewed'),
        ('declined', 'Declined'),
    ]

    scan        = models.ForeignKey(ScanRecord, on_delete=models.CASCADE, related_name='opinion_requests')
    patient     = models.ForeignKey(User, on_delete=models.CASCADE, related_name='opinion_requests_sent')
    doctor      = models.ForeignKey(User, on_delete=models.CASCADE, related_name='opinion_requests_received')
    status      = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    message     = models.TextField(blank=True)       # patient's message to doctor
    doctor_note = models.TextField(blank=True)        # doctor's review response
    urgency     = models.CharField(
        max_length=10,
        choices=[('low','Low'),('normal','Normal'),('urgent','Urgent')],
        default='normal'
    )
    created_at  = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient.username} → Dr.{self.doctor.username} [{self.status}]"

    def mark_reviewed(self, note):
        self.doctor_note = note
        self.status      = 'reviewed'
        self.reviewed_at = timezone.now()
        self.save()


# ═══════════════════════════════════════════════
# BONUS — Patient Vitals Tracker
# ═══════════════════════════════════════════════

class PatientVitals(models.Model):
    patient         = models.ForeignKey(User, on_delete=models.CASCADE, related_name='vitals')
    recorded_at     = models.DateTimeField(auto_now_add=True)
    temperature     = models.FloatField(null=True, blank=True, help_text='°C')
    spo2            = models.FloatField(null=True, blank=True, help_text='Oxygen saturation %')
    heart_rate      = models.PositiveIntegerField(null=True, blank=True, help_text='BPM')
    respiratory_rate= models.PositiveIntegerField(null=True, blank=True, help_text='Breaths/min')
    blood_pressure  = models.CharField(max_length=20, blank=True, help_text='e.g. 120/80')
    notes           = models.TextField(blank=True)

    class Meta:
        ordering = ['-recorded_at']

    def __str__(self):
        return f"{self.patient.username} vitals @ {self.recorded_at:%Y-%m-%d %H:%M}"

    @property
    def spo2_status(self):
        if not self.spo2:
            return 'unknown'
        if self.spo2 >= 95:   return 'normal'
        if self.spo2 >= 90:   return 'low'
        return 'critical'


# ═══════════════════════════════════════════════
# BONUS — Secure Report Sharing Link
# ═══════════════════════════════════════════════

class ReportShare(models.Model):
    record      = models.ForeignKey(ScanRecord, on_delete=models.CASCADE, related_name='shares')
    share_token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    created_by  = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at  = models.DateTimeField(auto_now_add=True)
    expires_at  = models.DateTimeField()
    accessed    = models.BooleanField(default=False)
    access_count= models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Share {self.share_token} — expires {self.expires_at:%Y-%m-%d}"

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at

    @property
    def share_url(self):
        return f"/report/share/{self.share_token}/"


# ═══════════════════════════════════════════════
# EXISTING — Diabetes Assessment
# ═══════════════════════════════════════════════

class DiabetesAssessment(models.Model):
    Age = models.IntegerField()
    Gender = models.IntegerField()
    Polyuria = models.CharField(max_length=10)
    Polydipsia = models.CharField(max_length=10)
    sudden_weight_loss = models.CharField(max_length=10)
    weakness = models.CharField(max_length=10)
    Polyphagia = models.CharField(max_length=10)
    Genital_thrush = models.CharField(max_length=10)
    visual_blurring = models.CharField(max_length=10)
    Itching = models.CharField(max_length=10)
    Irritability = models.CharField(max_length=10)
    delayed_healing = models.CharField(max_length=10)
    partial_paresis = models.CharField(max_length=10)
    muscle_stiffness = models.CharField(max_length=10)
    Alopecia = models.CharField(max_length=10)
    Obesity = models.CharField(max_length=10)
    label = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f'{self.label}'