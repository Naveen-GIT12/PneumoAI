"""
App/urls.py — All URL patterns for PneumoAI
"""
from django.contrib import admin
from django.urls import path
from .views import (
    home, RegisterView, PatientRegisterView, DoctorRegisterView,
    CustomLoginView, OTPVerifyView, EmailVerificationView,
    logout_view, profile,
    Deploy_8, Database,
    # Scan history
    scan_history, scan_detail, download_pdf, email_report,
    # Feature 1 — Analytics
    analytics,
    # Feature 2 — Doctor-Patient
    my_patients, assign_patient, unassign_patient, patient_scans,
    # Feature 3 — Second Opinion
    request_second_opinion, my_opinion_requests, review_opinion,
    # Feature 7 — Comparison
    compare_scans,
    # Feature 8 — Admin Dashboard
    admin_dashboard,
    # Bonus
    vitals, create_share_link, view_shared_report,
)
from . import views

urlpatterns = [

    # Home
    path('', home, name='users-home'),
    path('profile/', home, name='users-home'),

    # Registration
    path('register/', PatientRegisterView.as_view(), name='users-register'),
    path('register/patient/', PatientRegisterView.as_view(), name='patient-register'),
    path('register/doctor/', DoctorRegisterView.as_view(), name='doctor-register'),

    # Authentication
    path('login/', CustomLoginView.as_view(), name='login'),
    path('verify-email/<str:token>/', EmailVerificationView.as_view(), name='verify-email'),
    path('verify-otp/', OTPVerifyView.as_view(), name='verify-otp'),
    path('logout_view/', logout_view, name='logout_view'),

    # Profile
    path('profile1/', views.profile, name='users-profile'),

    # Core App
    path('Deploy_8/', Deploy_8, name='Deploy_8'),
    path('Database/', Database, name='Database'),

    # Scan History
    path('history/', scan_history, name='scan-history'),
    path('history/<int:pk>/', scan_detail, name='scan-detail'),
    path('history/<int:pk>/pdf/', download_pdf, name='download-pdf'),
    path('history/<int:pk>/email/', email_report, name='email-report'),

    # Analytics Dashboard
    path('analytics/', analytics, name='analytics'),

    # Doctor-Patient Assignment
    path('patients/', my_patients, name='my-patients'),
    path('patients/assign/', assign_patient, name='assign-patient'),
    path('patients/unassign/<int:pk>/', unassign_patient, name='unassign-patient'),
    path('patients/<int:patient_pk>/scans/', patient_scans, name='patient-scans'),

    # Second Opinion
    path('second-opinion/<int:scan_pk>/', request_second_opinion, name='request-opinion'),
    path('second-opinion/requests/', my_opinion_requests, name='opinion-requests'),
    path('second-opinion/review/<int:req_pk>/', review_opinion, name='review-opinion'),

    # Comparison
    path('compare/', compare_scans, name='compare-scans'),

    # Admin Dashboard
    path('admin-dashboard/', admin_dashboard, name='admin-dashboard'),

    # Vitals Tracker
    path('vitals/', vitals, name='vitals'),

    # Secure Share Link
    path('history/<int:pk>/share/', create_share_link, name='create-share'),
    path('report/share/<uuid:token>/', view_shared_report, name='view-shared'),
]