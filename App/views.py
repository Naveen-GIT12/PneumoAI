"""
App/views.py
─────────────────────────────────────────────────────────────
ALL views for PneumoAI including all 8 new features:
  1. Trend Analytics Dashboard
  2. Doctor–Patient Assignment
  3. Second Opinion Request
  4. Email Report Delivery
  5. Severity Scoring
  6. Gemini Treatment Recommendations
  7. Comparison View
  8. Admin Dashboard
  + Bonus: Vitals Tracker, Secure Share Link
─────────────────────────────────────────────────────────────
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout, login as auth_login
from django.contrib.auth.models import User
from django.conf import settings
from django.core.mail import send_mail, EmailMessage
from django.http import HttpResponse, JsonResponse
from django.core.exceptions import PermissionDenied
from django.db.models import Count, Avg, Q
from django.db.models.functions import TruncMonth, TruncWeek
from django.utils import timezone

import os, re, json, random, uuid, io
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tensorflow import keras

from .forms import (
    PatientRegisterForm, DoctorRegisterForm, LoginForm, OTPForm,
    UpdateUserForm, UpdateProfileForm, UserImageForm,
    PatientScanForm, DoctorScanForm,
    SecondOpinionForm, VitalsForm,
)
from .models import (
    UserImageModel, Profile, DoctorProfile, ScanRecord,
    DoctorPatientLink, SecondOpinionRequest, PatientVitals, ReportShare,
)
from . import forms

import google.generativeai as genai
from .gradcam import generate_advanced_heatmap, describe_from_boxes_enhanced

# ─────────────────────────────────────────
# Configure Gemini
# ─────────────────────────────────────────
genai.configure(api_key=settings.GEMINI_API_KEY)

# Keep one inference model in memory instead of loading Model.h5 for every scan.
# compile=False avoids loading training/optimizer state that inference does not need.
@lru_cache(maxsize=1)
def get_prediction_model():
    model_path = os.path.join(settings.BASE_DIR, 'App', 'Model.h5')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"PneumoAI model not found: {model_path}")
    print("[PneumoAI] Loading prediction model into memory...")
    model = keras.models.load_model(model_path, compile=False)
    print("[PneumoAI] Prediction model loaded and cached.")
    return model


# Gemini recommendations are not required to calculate the diagnosis.
# Run them after the result page is returned so external API latency does not
# hold up the scan response. Use a small pool to avoid creating unlimited threads.
_RECOMMENDATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="pneumoai-recommendations"
)


def _generate_recommendations_for_record(record_id, pred_class, age, gender, notes):
    try:
        recommendations = get_gemini_recommendations(
            pred_class, age, gender, notes
        )
        if recommendations:
            ScanRecord.objects.filter(pk=record_id).update(
                treatment_recs=json.dumps(recommendations)
            )
            print(f"[PneumoAI] Gemini recommendations saved for scan {record_id}.")
        else:
            print(f"[PneumoAI] Gemini returned no recommendations for scan {record_id}.")
    except Exception as exc:
        print(f"[PneumoAI] Background recommendation error for scan {record_id}: {exc}")


def _prediction_probabilities(prediction):
    """Return a valid probability vector without changing already-softmaxed output."""
    values = np.asarray(prediction, dtype=np.float32).reshape(-1)
    if values.size != 5 or not np.all(np.isfinite(values)):
        raise ValueError("The model must return five finite class scores.")
    if np.all(values >= 0) and np.all(values <= 1) and np.isclose(values.sum(), 1.0, atol=1e-3):
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)

LABEL_MAP = {
    "COVID_19":            "COVID-19",
    "normal":              "Normal",
    "opacity":             "Lung Opacity",
    "Pneumonia_Bacterial": "Bacterial Pneumonia",
    "Pneumonia_Viral":     "Viral Pneumonia",
}
CONFIDENCE_MAP = {
    "COVID_19": 88.0, "normal": 98.0,
    "Pneumonia_Bacterial": 92.0, "Pneumonia_Viral": 90.0, "opacity": 85.0,
}


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def send_login_notification(user):
    try:
        send_mail(
            subject='New Login — PneumoAI',
            message=(f'Hello {user.get_full_name() or user.username},\n\n'
                     f'A new login was detected for your account ({user.username}).\n'
                     f'If this was not you, reset your password immediately.\n\n— PneumoAI'),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email], fail_silently=True,
        )
    except Exception as e:
        print(f"Login email error: {e}")


def send_otp_email(user, otp):
    try:
        send_mail(
            subject='Your OTP — PneumoAI',
            message=(f'Hello {user.get_full_name() or user.username},\n\n'
                     f'Your OTP is: {otp}\n\nValid for 5 minutes.\n\n— PneumoAI'),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email], fail_silently=True,
        )
    except Exception as e:
        print(f"OTP email error: {e}")


def send_verification_email(request, user):
    """Send a signed email-verification link valid for 24 hours."""
    try:
        signer = TimestampSigner(salt="pneumoai-email-verification")
        token = signer.sign(str(user.pk))
        verification_url = request.build_absolute_uri(
            reverse_lazy("verify-email", kwargs={"token": token})
        )
        sent = send_mail(
            subject="Verify your PneumoAI email address",
            message=(
                f"Hello {user.get_full_name() or user.username},\\n\\n"
                "Thank you for registering with PneumoAI.\\n"
                "Verify your email address using this link:\\n\\n"
                f"{verification_url}\\n\\n"
                "This link expires in 24 hours.\\n\\n"
                "— PneumoAI"
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        return sent == 1
    except Exception as exc:
        print(f"Email verification error: {exc}")
        return False


def extract_json(text):
    try:
        match = re.search(r'\[.*\]', text, re.S)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


def extract_json_object(text):
    """Extract first {...} JSON object from Gemini response."""
    try:
        match = re.search(r'\{.*\}', text, re.S)
        if match:
            return match.group()
    except Exception:
        pass
    return '{}'


def describe_from_boxes(boxes):
    if not boxes:
        return "No significant abnormal regions identified. Both lungs appear clear."
    regions = []
    for b in boxes:
        cx = ((b.get('x1', .3) + b.get('x2', .7)) / 2)
        cy = ((b.get('y1', .4) + b.get('y2', .7)) / 2)
        side  = 'left' if cx < .5 else 'right'
        level = 'upper' if cy < .33 else ('mid' if cy < .66 else 'lower')
        regions.append(f"{side} {level} lung")
    return (f"AI analysis identified abnormal activity in the "
            f"{', '.join(sorted(set(regions)))}. "
            f"Highlighted regions show increased radiodensity consistent with the detected condition. "
            f"Please correlate with clinical findings.")


def calculate_severity(pred_class, confidence, boxes):
    """Feature 5 — Severity scoring."""
    if pred_class == 'normal':
        return 'None', 0.0
    total_area = sum(
        (b.get('x2', .7) - b.get('x1', .3)) * (b.get('y2', .7) - b.get('y1', .4))
        for b in boxes
    )
    area_score  = min(total_area / 0.5, 1.0)
    score       = (confidence / 100 * 0.6) + (area_score * 0.4)
    if score >= 0.75:   sev = 'Severe'
    elif score >= 0.45: sev = 'Moderate'
    else:               sev = 'Low'
    return sev, round(area_score, 3)


def get_gemini_recommendations(pred_class, patient_age=None, patient_gender=None, clinical_notes=None):
    """Feature 6 — Gemini-powered treatment recommendations."""
    try:
        gemini = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
A patient has been diagnosed with {pred_class.replace('_', ' ')} via chest X-ray AI analysis.

Patient details:
- Age: {patient_age or 'Not provided'}
- Gender: {patient_gender or 'Not provided'}
- Clinical notes: {clinical_notes or 'None provided'}

Provide a concise clinical response as JSON only (no markdown, no explanation):
{{
  "immediate": ["action 1", "action 2", "action 3"],
  "medications": ["med 1 (dose)", "med 2 (dose)"],
  "lifestyle": ["advice 1", "advice 2", "advice 3"],
  "emergency_signs": ["sign 1", "sign 2"],
  "followup": "Follow-up timeline string"
}}

Keep each point under 12 words. Generic medication names only.
"""
        response = gemini.generate_content(prompt)
        if response and response.text:
            raw = extract_json_object(response.text)
            return json.loads(raw)
    except Exception as e:
        print(f"Gemini recommendations error: {e}")
    return None


# generate_heatmap removed — replaced by generate_advanced_heatmap from gradcam.py


def generate_pdf_buffer(record, request=None):
    """
    Generate a clean, structured PDF report for a PneumoAI ScanRecord.
    Used by both download_pdf() and email_report(); existing routes remain unchanged.
    """
    import io
    from datetime import datetime
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether
    )
    from reportlab.lib.utils import ImageReader

    def safe(value):
        if value is None or value == "":
            return "Not provided"
        return escape(str(value)).replace("\n", "<br/>")

    def make_paragraph(value, style):
        return Paragraph(safe(value), style)

    buffer = io.BytesIO()

    NAVY = colors.HexColor("#14243A")
    BLUE = colors.HexColor("#1677B8")
    PALE = colors.HexColor("#EEF5FA")
    LIGHT = colors.HexColor("#F7FAFC")
    TEXT = colors.HexColor("#273444")
    MUTED = colors.HexColor("#64748B")
    BORDER = colors.HexColor("#D9E2EA")
    RED = colors.HexColor("#B42318")
    ORANGE = colors.HexColor("#B85C00")
    GREEN = colors.HexColor("#18794E")
    PURPLE = colors.HexColor("#6941C6")

    class NumberedCanvas:
        """Page-number canvas wrapper is intentionally omitted; use simple footer below."""

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PneumoTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, textColor=NAVY, alignment=0, spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        "PneumoSubtitle", parent=styles["Normal"], fontSize=8.5,
        leading=12, textColor=MUTED, spaceAfter=8
    )
    section_style = ParagraphStyle(
        "PneumoSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "PneumoBody", parent=styles["BodyText"], fontSize=8.5,
        leading=12, textColor=TEXT, spaceAfter=4
    )
    label_style = ParagraphStyle(
        "PneumoLabel", parent=body_style, fontSize=7.2,
        leading=9, textColor=MUTED, fontName="Helvetica-Bold", spaceAfter=1
    )
    value_style = ParagraphStyle(
        "PneumoValue", parent=body_style, fontSize=8.8,
        leading=11, textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=0
    )
    small_style = ParagraphStyle(
        "PneumoSmall", parent=body_style, fontSize=7.2,
        leading=9.5, textColor=MUTED
    )
    center_style = ParagraphStyle(
        "PneumoCenter", parent=value_style, alignment=TA_CENTER
    )
    result_style = ParagraphStyle(
        "PneumoResult", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER, textColor=BLUE
    )
    disclaimer_style = ParagraphStyle(
        "PneumoDisclaimer", parent=body_style, fontSize=7.6,
        leading=10.5, textColor=TEXT
    )

    class FooterCanvas:
        """Small page footer using a normal ReportLab canvas callback."""
        pass

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(1.8*cm, 1.25*cm, A4[0] - 1.8*cm, 1.25*cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.8*cm, 0.85*cm, "PneumoAI · AI-assisted chest X-ray analysis")
        canvas.drawRightString(A4[0] - 1.8*cm, 0.85*cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.45*cm, bottomMargin=1.75*cm,
        title="PneumoAI Medical Diagnostic Support Report",
        author="PneumoAI Detection System"
    )

    label_map = {
        "COVID_19": "COVID-19",
        "normal": "Normal",
        "opacity": "Lung Opacity",
        "Pneumonia_Bacterial": "Bacterial Pneumonia",
        "Pneumonia_Viral": "Viral Pneumonia",
    }
    predicted_class = getattr(record, "predicted_class", "")
    predicted_label = getattr(record, "predicted_label", None) or label_map.get(
        predicted_class, predicted_class.replace("_", " ").title()
    )
    result_color = {
        "COVID_19": RED,
        "normal": GREEN,
        "opacity": PURPLE,
        "Pneumonia_Bacterial": ORANGE,
        "Pneumonia_Viral": RED,
    }.get(predicted_class, BLUE)

    created_at = getattr(record, "created_at", None)
    if created_at:
        report_date = created_at.strftime("%B %d, %Y")
        report_id = f"MD-{created_at.strftime('%Y%m%d')}-{record.pk:04d}"
    else:
        report_date = datetime.now().strftime("%B %d, %Y")
        report_id = f"MD-{getattr(record, 'pk', 'N/A')}"

    try:
        confidence_value = float(record.confidence)
        confidence_text = f"{confidence_value:.1f}%"
    except (TypeError, ValueError, AttributeError):
        confidence_value = 0.0
        confidence_text = "Not available"

    severity = getattr(record, "severity", None) or "Not assessed"
    severity_color = {
        "None": GREEN, "Low": BLUE, "Moderate": ORANGE, "Severe": RED
    }.get(severity, MUTED)

    try:
        area_value = float(getattr(record, "heatmap_area", 0) or 0)
        area_text = f"{area_value * 100:.1f}%"
    except (TypeError, ValueError):
        area_text = "Not available"

    story = []

    # Header
    header = Table([[
        Paragraph("<b>PNEUMOAI</b>", ParagraphStyle(
            "PneumoBrand", parent=title_style, fontSize=12, textColor=BLUE
        )),
        Paragraph("AI-ASSISTED CHEST X-RAY REPORT", ParagraphStyle(
            "PneumoHeaderRight", parent=small_style, alignment=TA_RIGHT
        ))
    ]], colWidths=[8.7*cm, 8.7*cm])
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 1.5, BLUE),
    ]))
    story += [
        header, Spacer(1, 7),
        Paragraph("Medical Diagnostic Support Report", title_style),
        Paragraph(
            "Deep-learning image classification summary · For qualified clinician review",
            subtitle_style
        )
    ]

    # Report metadata
    meta = Table([
        [make_paragraph("REPORT DATE", label_style),
         make_paragraph("REPORT ID", label_style),
         make_paragraph("MODEL", label_style),
         make_paragraph("STATUS", label_style)],
        [make_paragraph(report_date, value_style),
         make_paragraph(report_id, value_style),
         make_paragraph("PneumoAI v2", value_style),
         make_paragraph("Analysis complete", value_style)]
    ], colWidths=[4.35*cm] * 4)
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [meta]

    # Patient and reporting-doctor details for doctor scans
    if getattr(record, "is_doctor_scan", False):
        story.append(Paragraph("1. Patient Information", section_style))
        patient_rows = []
        patient_fields = [
            ("Patient name", getattr(record, "patient_name", None)),
            ("Patient ID", getattr(record, "patient_id", None)),
            ("Age", f"{record.patient_age} years" if getattr(record, "patient_age", None) else None),
            ("Gender", getattr(record, "patient_gender", None)),
            ("X-ray date", getattr(record, "xray_date", None)),
            ("Clinical notes", getattr(record, "clinical_notes", None)),
        ]
        for caption, val in patient_fields:
            if val not in (None, ""):
                if caption == "X-ray date" and hasattr(val, "strftime"):
                    val = val.strftime("%B %d, %Y")
                patient_rows.append([
                    make_paragraph(caption, label_style),
                    make_paragraph(val, body_style)
                ])
        try:
            doctor_profile = record.user.doctorprofile
            doctor_name = record.user.get_full_name()
            if doctor_name or getattr(doctor_profile, "specialization", None):
                patient_rows.append([
                    make_paragraph("Reporting doctor", label_style),
                    make_paragraph(
                        f"Dr. {doctor_name} · {doctor_profile.specialization}", body_style
                    )
                ])
            for caption, attr in [
                ("Hospital", "hospital_name"),
                ("Registration number", "doctor_registration_id")
            ]:
                val = getattr(doctor_profile, attr, None)
                if val:
                    patient_rows.append([
                        make_paragraph(caption, label_style),
                        make_paragraph(val, body_style)
                    ])
        except Exception:
            pass

        if patient_rows:
            patient_table = Table(patient_rows, colWidths=[4.0*cm, 13.4*cm])
            patient_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
            ]))
            story.append(patient_table)

    # Classification result
    story.append(Paragraph("2. AI Classification Result", section_style))
    result_card = Table([
        [Paragraph("MODEL-PREDICTED CLASS", ParagraphStyle(
            "ResultLabel", parent=label_style, alignment=TA_CENTER
        ))],
        [Paragraph(safe(predicted_label).upper(), ParagraphStyle(
            "ResultValue", parent=result_style, textColor=result_color
        ))],
        [Paragraph(f"Model confidence: {confidence_text}", center_style)]
    ], colWidths=[17.4*cm])
    result_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
        ("LINEABOVE", (0, 0), (-1, 0), 3, result_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(result_card)
    story.append(Spacer(1, 6))

    metrics = Table([
        [make_paragraph("SEVERITY LABEL", label_style),
         make_paragraph("REPORTED AFFECTED AREA", label_style)],
        [Paragraph(safe(severity), ParagraphStyle(
            "SeverityValue", parent=value_style, textColor=severity_color
        )), make_paragraph(area_text, value_style)]
    ], colWidths=[8.7*cm, 8.7*cm])
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(metrics)

    # X-ray and Grad-CAM heatmap, if stored and readable
    image_paths = []
    for field_name, caption in [
        ("image", "Uploaded chest X-ray"),
        ("heatmap_image", "AI Grad-CAM explanation")
    ]:
        field = getattr(record, field_name, None)
        try:
            if field and getattr(field, "path", None):
                import os
                if os.path.isfile(field.path):
                    image_paths.append((field.path, caption))
        except Exception:
            pass

    if image_paths:
        story.append(Paragraph("3. Image Review", section_style))
        image_cells = []
        for path, caption in image_paths:
            try:
                img = RLImage(path)
                img._restrictSize(7.7*cm, 7.0*cm)
                image_cells.append([
                    img,
                    Paragraph(safe(caption), ParagraphStyle(
                        "ImageCaption", parent=small_style, alignment=TA_CENTER
                    ))
                ])
            except Exception:
                continue
        if image_cells:
            row = []
            for img, caption in image_cells:
                row.append([img, caption])
            image_table = Table([row], colWidths=[8.7*cm] * len(row))
            image_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(image_table)

    # Interpretation
    story.append(Paragraph("4. AI Interpretation", section_style))
    interpretation = getattr(record, "affected_text", None) or (
        "The model completed image classification. No additional interpretation was stored."
    )
    story.append(Paragraph(safe(interpretation), body_style))
    story.append(Paragraph(
        "The model confidence is a model score, not the probability that the patient has "
        "the predicted condition. The affected-area value is an automated estimate and "
        "is not a validated measurement of lung involvement.",
        small_style
    ))

    # Recommendations (retain the existing stored content, without fabricating any)
    try:
        recommendations = record.get_treatment_recs()
    except Exception:
        recommendations = None

    if isinstance(recommendations, dict):
        rec_sections = [
            ("immediate", "General actions"),
            ("medications", "Medication-related output"),
            ("lifestyle", "Lifestyle guidance"),
            ("emergency_signs", "Warning signs"),
            ("followup", "Follow-up"),
        ]
        rec_rows = []
        for key, heading in rec_sections:
            value = recommendations.get(key)
            if not value:
                continue
            if isinstance(value, (list, tuple)):
                value_text = "<br/>".join("• " + safe(item) for item in value)
                value_para = Paragraph(value_text, body_style)
            else:
                value_para = Paragraph(safe(value), body_style)
            rec_rows.append([make_paragraph(heading, label_style), value_para])

        if rec_rows:
            story.append(Paragraph("5. AI-Generated Care Information", section_style))
            story.append(Paragraph(
                "The following is AI-generated informational content, not a prescription. "
                "A qualified clinician must verify it before any action is taken.",
                small_style
            ))
            rec_table = Table(rec_rows, colWidths=[4.0*cm, 13.4*cm], repeatRows=0)
            rec_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
            ]))
            story.append(rec_table)

    # Medical disclaimer
    story += [
        Spacer(1, 8),
        Paragraph("Important Limitations & Medical Disclaimer", section_style),
    ]
    disclaimer = Table([[
        Paragraph(
            "<b>IMPORTANT:</b> This AI-assisted report is for informational and research "
            "decision-support purposes only. It is not a definitive diagnosis, radiology "
            "report, or treatment prescription. AI predictions may be incorrect and must "
            "not be used alone to diagnose, exclude a condition, or make treatment decisions. "
            "A qualified healthcare professional must review the original image and relevant "
            "clinical information. Seek urgent medical attention for severe or worsening symptoms.",
            disclaimer_style
        )
    ]], colWidths=[17.4*cm])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7ED")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#F2C078")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [
        disclaimer, Spacer(1, 6),
        Paragraph(
            f"© {datetime.now().year} PneumoAI Detection System · "
            f"Report ID: {safe(report_id)} · Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            ParagraphStyle("PneumoCopyright", parent=small_style, alignment=TA_CENTER)
        )
    ]

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    buffer.seek(0)
    return buffer



# ═══════════════════════════════════════════════
# AUTH VIEWS
# ═══════════════════════════════════════════════

def home(request):
    return render(request, 'users/home.html')


def index(request):
    return render(request, 'app/index.html')


class PatientRegisterView(View):
    template_name = 'users/patient_register.html'
    def get(self, request):
        return render(request, self.template_name, {'form': PatientRegisterForm()})
    def post(self, request):
        form = PatientRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created. Please log in.')
            return redirect('login')
        return render(request, self.template_name, {'form': form})


class DoctorRegisterView(View):
    template_name = 'users/doctor_register.html'
    def get(self, request):
        return render(request, self.template_name, {'form': DoctorRegisterForm()})
    def post(self, request):
        form = DoctorRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Doctor account created. Please log in.')
            return redirect('login')
        return render(request, self.template_name, {'form': form})


class EmailVerificationView(View):
    """Activate a newly registered account after its email is verified."""

    def get(self, request, token):
        signer = TimestampSigner(salt="pneumoai-email-verification")
        try:
            user_id = signer.unsign(token, max_age=24 * 60 * 60)
        except SignatureExpired:
            messages.error(
                request,
                "This verification link has expired. Please contact support for assistance.",
            )
            return redirect("login")
        except BadSignature:
            messages.error(request, "This verification link is invalid.")
            return redirect("login")

        try:
            user = User.objects.get(pk=user_id)
        except (User.DoesNotExist, ValueError, TypeError):
            messages.error(
                request,
                "The account associated with this link could not be found.",
            )
            return redirect("login")

        if user.is_active:
            messages.info(request, "Your email is already verified. You can log in.")
            return redirect("login")

        user.is_active = True
        user.save(update_fields=["is_active"])
        messages.success(
            request,
            "Your email has been verified successfully. You can now log in.",
        )
        return redirect("login")


class RegisterView(PatientRegisterView):
    pass


class CustomLoginView(LoginView):
    form_class    = LoginForm
    template_name = 'users/login.html'
    def form_valid(self, form):
        user = form.get_user()
        otp  = str(random.randint(100000, 999999))
        self.request.session['otp']         = otp
        self.request.session['otp_user_id'] = user.id
        self.request.session['otp_created'] = datetime.now().isoformat()
        self.request.session['remember_me'] = form.cleaned_data.get('remember_me', False)
        send_otp_email(user, otp)
        messages.info(self.request, f'OTP sent to {user.email}.')
        return redirect('verify-otp')


class OTPVerifyView(View):
    template_name = 'users/otp_verify.html'
    def get(self, request):
        if 'otp_user_id' not in request.session:
            return redirect('login')
        return render(request, self.template_name, {'form': OTPForm()})
    def post(self, request):
        if 'otp_user_id' not in request.session:
            return redirect('login')
        form = OTPForm(request.POST)
        if form.is_valid():
            entered     = form.cleaned_data['otp']
            session_otp = request.session.get('otp')
            user_id     = request.session.get('otp_user_id')
            otp_created = request.session.get('otp_created')
            remember_me = request.session.get('remember_me', False)
            if otp_created:
                elapsed = (datetime.now() - datetime.fromisoformat(otp_created)).total_seconds()
                if elapsed > 300:
                    messages.error(request, 'OTP expired. Please log in again.')
                    return redirect('login')
            if entered == session_otp:
                user = User.objects.get(id=user_id)
                auth_login(request, user, backend='django.contrib.auth.backends.ModelBackend')
                if not remember_me:
                    request.session.set_expiry(0)
                send_login_notification(user)
                for k in ['otp', 'otp_user_id', 'otp_created', 'remember_me']:
                    request.session.pop(k, None)
                messages.success(request, f'Welcome back, {user.get_full_name() or user.username}!')
                return redirect('/')
            else:
                messages.error(request, 'Invalid OTP. Please try again.')
        return render(request, self.template_name, {'form': form})


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name         = 'users/password_reset.html'
    email_template_name   = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message       = "Password reset email sent."
    success_url           = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name   = 'users/change_password.html'
    success_message = "Password changed successfully."
    success_url     = reverse_lazy('users-home')


@login_required
def profile(request):
    user = request.user
    if not hasattr(user, 'profile'):
        Profile.objects.create(user=user)
    if request.method == 'POST':
        uf = UpdateUserForm(request.POST, instance=user)
        pf = UpdateProfileForm(request.POST, request.FILES, instance=user.profile)
        if uf.is_valid() and pf.is_valid():
            uf.save(); pf.save()
            messages.success(request, 'Profile updated successfully.')
            return redirect('users-profile')
    else:
        uf = UpdateUserForm(instance=user)
        pf = UpdateProfileForm(instance=user.profile)
    return render(request, 'users/profile.html', {'user_form': uf, 'profile_form': pf})


# ═══════════════════════════════════════════════
# MAIN SCAN VIEW — Role-aware
# ═══════════════════════════════════════════════

@login_required
def Deploy_8(request):
    is_doctor = hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'
    FormClass = DoctorScanForm if is_doctor else PatientScanForm
    template  = 'App/model_doctor.html' if is_doctor else 'App/model.html'

    if request.method == 'POST':
        form = FormClass(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            ext        = image_file.name.rsplit('.', 1)[-1].lower()
            fname      = f"{uuid.uuid4().hex}.{ext}"
            save_dir   = os.path.join(settings.MEDIA_ROOT, 'images')
            os.makedirs(save_dir, exist_ok=True)
            save_path  = os.path.join(save_dir, fname)
            with open(save_path, 'wb+') as dest:
                for chunk in image_file.chunks():
                    dest.write(chunk)
            image_rel = f"images/{fname}"
            image_pil = Image.open(save_path).convert('RGB')

            # ── Fast model inference (cached model; no repeated disk loading)
            model = get_prediction_model()
            img_r = ImageOps.fit(image_pil, (224, 224), Image.Resampling.LANCZOS)
            arr = (np.asarray(img_r, dtype=np.float32) / 127.0) - 1.0
            prediction = model.predict(np.expand_dims(arr, axis=0), verbose=0)
            probabilities = _prediction_probabilities(prediction)
            classes = [
                'COVID_19', 'normal', 'opacity',
                'Pneumonia_Bacterial', 'Pneumonia_Viral'
            ]
            class_index = int(np.argmax(probabilities))
            pred_class = classes[class_index]
            pred_label = LABEL_MAP[pred_class]
            confidence = float(probabilities[class_index] * 100.0)

            # Do not make a second Gemini image-analysis request for bounding boxes.
            # Genuine Grad-CAM is generated from the prediction model below.
            boxes = []
            affected_text = describe_from_boxes_enhanced(boxes, pred_class)

            # Severity is retained. With no external bounding-box detector,
            # the affected-area component remains zero rather than being fabricated.
            severity, heatmap_area = calculate_severity(
                pred_class, confidence, boxes
            )

            # ── TRUE GRAD-CAM HEATMAP (with condition-specific colourmap)
            uid = uuid.uuid4().hex[:8]
            class_idx = class_index

            affected_url = generate_advanced_heatmap(
                image_pil   = image_pil,
                pred_class  = pred_class,
                confidence  = confidence,
                severity    = severity,
                uid         = uid,
                media_root  = settings.MEDIA_ROOT,
                media_url   = settings.MEDIA_URL,
                model       = model,
                boxes       = boxes,
                class_index = class_idx,
            )
            # Verify that Grad-CAM actually wrote the PNG before storing its path.
            # generate_advanced_heatmap returns: MEDIA_URL + 'heatmap/<filename>'.
            hm_rel = ''
            if affected_url:
                media_prefix = settings.MEDIA_URL.rstrip('/') + '/'
                relative_candidate = affected_url

                if relative_candidate.startswith(media_prefix):
                    relative_candidate = relative_candidate[len(media_prefix):]
                else:
                    # Accept a media-relative path, but never an arbitrary URL/path.
                    relative_candidate = relative_candidate.lstrip('/')

                relative_candidate = relative_candidate.replace('\\\\', '/')
                candidate_path = os.path.abspath(
                    os.path.join(settings.MEDIA_ROOT, relative_candidate)
                )
                media_root_abs = os.path.abspath(settings.MEDIA_ROOT)

                try:
                    is_inside_media = (
                        os.path.commonpath([media_root_abs, candidate_path])
                        == media_root_abs
                    )
                except ValueError:
                    is_inside_media = False

                if is_inside_media and os.path.isfile(candidate_path):
                    hm_rel = os.path.relpath(
                        candidate_path, media_root_abs
                    ).replace('\\\\', '/')
                    affected_url = settings.MEDIA_URL.rstrip('/') + '/' + hm_rel
                    print(f"[GradCAM] Heatmap verified: {candidate_path}")
                else:
                    affected_url = ''
                    print(
                        "[GradCAM] Returned URL does not point to an existing "
                        f"file inside MEDIA_ROOT: {candidate_path}"
                    )
            else:
                print("[GradCAM] Generation returned an empty URL.")

            # ── Feature 6: Gemini recommendations (for doctor scans or when age available)
            patient_age    = form.cleaned_data.get('patient_age')    if is_doctor else None
            patient_gender = form.cleaned_data.get('patient_gender') if is_doctor else None
            clinical_notes_val = form.cleaned_data.get('clinical_notes', '') if is_doctor else ''

            # Recommendations are generated after the result page is returned.
            # This prevents Gemini network latency from blocking diagnosis display.
            treatment_recs = None

            # ── Build ScanRecord
            rk = dict(
                user            = request.user,
                predicted_class = pred_class,
                predicted_label = pred_label,
                confidence      = confidence,
                affected_text   = affected_text,
                severity        = severity,
                heatmap_area    = heatmap_area,
                treatment_recs  = treatment_recs,
            )
            if is_doctor:
                rk.update(
                    patient_id     = form.cleaned_data['patient_id'],
                    patient_name   = form.cleaned_data['patient_name'],
                    patient_age    = form.cleaned_data['patient_age'],
                    patient_gender = form.cleaned_data['patient_gender'],
                    xray_date      = form.cleaned_data['xray_date'],
                    clinical_notes = clinical_notes_val,
                )
            record = ScanRecord(**rk)
            record.image.name = image_rel
            record.save()
            if hm_rel:
                record.heatmap_image.name = hm_rel
                record.save(update_fields=['heatmap_image'])

            # Preserve the legacy image record.
            old = UserImageModel()
            old.label = pred_class
            old.image.name = image_rel
            old.save()

            # Generate Gemini recommendations in the background.
            if pred_class != 'normal':
                _RECOMMENDATION_EXECUTOR.submit(
                    _generate_recommendations_for_record,
                    record.pk, pred_class, patient_age,
                    patient_gender, clinical_notes_val
                )

            return render(request, 'App/output.html', {
                'record': record, 'obj': old,
                'predict': pred_class, 'predicted': pred_label,
                'confidence': confidence, 'severity': severity,
                'affected_image': affected_url, 'affected_text': affected_text,
                'treatment_recs': record.get_treatment_recs(),
                'is_doctor': is_doctor,
            })
    else:
        form = FormClass()

    dp = None
    if is_doctor:
        try: dp = request.user.doctorprofile
        except Exception: pass
    return render(request, template, {'form': form, 'is_doctor': is_doctor, 'doctor_profile': dp})


# ═══════════════════════════════════════════════
# SCAN HISTORY
# ═══════════════════════════════════════════════

@login_required
def scan_history(request):
    is_doctor = hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'
    records   = ScanRecord.objects.filter(user=request.user)

    # If doctor: also include scans of assigned patients
    if is_doctor:
        linked_patients = DoctorPatientLink.objects.filter(
            doctor=request.user, is_active=True
        ).values_list('patient_id', flat=True)
        records = ScanRecord.objects.filter(
            Q(user=request.user) | Q(user__in=linked_patients)
        )

    lf = request.GET.get('label', '')
    df = request.GET.get('date_from', '')
    dt = request.GET.get('date_to', '')
    sr = request.GET.get('search', '')
    sev= request.GET.get('severity', '')

    if lf:  records = records.filter(predicted_class=lf)
    if df:  records = records.filter(created_at__date__gte=df)
    if dt:  records = records.filter(created_at__date__lte=dt)
    if sev: records = records.filter(severity=sev)
    if sr and is_doctor:
        records = records.filter(
            Q(patient_name__icontains=sr) | Q(patient_id__icontains=sr)
        )

    return render(request, 'App/history.html', {
        'records': records, 'is_doctor': is_doctor,
        'label_filter': lf, 'date_from': df,
        'date_to': dt, 'search': sr, 'severity_filter': sev,
    })


@login_required
def scan_detail(request, pk):
    record = get_object_or_404(ScanRecord, pk=pk)
    # Allow doctor to view if patient is linked
    is_linked_doctor = (
        hasattr(request.user, 'profile') and
        request.user.profile.user_type == 'doctor' and
        DoctorPatientLink.objects.filter(
            doctor=request.user, patient=record.user, is_active=True
        ).exists()
    )
    if record.user != request.user and not is_linked_doctor:
        raise PermissionDenied
    au = ''
    try:
        if record.heatmap_image and record.heatmap_image.storage.exists(record.heatmap_image.name):
            au = record.heatmap_image.url
    except (ValueError, OSError):
        au = ''
    return render(request, 'App/output.html', {
        'record': record, 'obj': record,
        'predict': record.predicted_class, 'predicted': record.predicted_label,
        'confidence': record.confidence, 'severity': getattr(record, 'severity', 'None'),
        'affected_image': au, 'affected_text': record.affected_text,
        'treatment_recs': record.get_treatment_recs(),
        'is_doctor': record.is_doctor_scan,
    })


# ═══════════════════════════════════════════════
# FEATURE 4 — Download + Email PDF
# ═══════════════════════════════════════════════

@login_required
def download_pdf(request, pk):
    record = get_object_or_404(ScanRecord, pk=pk)
    if record.user != request.user:
        is_linked = DoctorPatientLink.objects.filter(
            doctor=request.user, patient=record.user, is_active=True).exists()
        if not is_linked:
            raise PermissionDenied
    try:
        buffer = generate_pdf_buffer(record, request)
        pname  = record.patient_name.replace(' ', '_') if record.patient_name else 'report'
        fname  = f"PneumoAI_{pname}_{record.predicted_class}.pdf"
        resp   = HttpResponse(buffer, content_type='application/pdf')
        resp['Content-Disposition'] = f'attachment; filename="{fname}"'
        return resp
    except ImportError:
        return HttpResponse("Install ReportLab: pip install reportlab", status=500)


@login_required
def email_report(request, pk):
    """Feature 4 — Email the PDF report to user."""
    record = get_object_or_404(ScanRecord, pk=pk)
    if record.user != request.user:
        raise PermissionDenied
    try:
        buffer = generate_pdf_buffer(record, request)
        pname  = record.patient_name or record.user.get_full_name() or record.user.username
        email  = EmailMessage(
            subject    = f'PneumoAI Diagnostic Report — {record.predicted_label}',
            body       = (f'Dear {record.user.get_full_name() or record.user.username},\n\n'
                          f'Please find your PneumoAI diagnostic report attached.\n\n'
                          f'Result    : {record.predicted_label}\n'
                          f'Severity  : {getattr(record, "severity", "N/A")}\n'
                          f'Confidence: {record.confidence:.1f}%\n'
                          f'Date      : {record.created_at.strftime("%B %d, %Y")}\n\n'
                          f'⚠ This is an AI-assisted report. Please consult a doctor.\n\n'
                          f'— PneumoAI Detection System'),
            from_email = settings.DEFAULT_FROM_EMAIL,
            to         = [record.user.email],
        )
        email.attach(f'PneumoAI_Report_{record.pk}.pdf', buffer.read(), 'application/pdf')
        email.send()
        messages.success(request, f'Report sent to {record.user.email} successfully!')
    except ImportError:
        messages.error(request, 'Install ReportLab: pip install reportlab')
    except Exception as e:
        messages.error(request, f'Failed to send email: {e}')
    return redirect('scan-detail', pk=pk)


# ═══════════════════════════════════════════════
# FEATURE 1 — Trend Analytics Dashboard
# ═══════════════════════════════════════════════

@login_required
def analytics(request):
    is_doctor = hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'

    if is_doctor:
        linked = DoctorPatientLink.objects.filter(
            doctor=request.user, is_active=True
        ).values_list('patient_id', flat=True)
        records = ScanRecord.objects.filter(Q(user=request.user) | Q(user__in=linked))
    else:
        records = ScanRecord.objects.filter(user=request.user)

    total      = records.count()
    avg_conf   = records.aggregate(Avg('confidence'))['confidence__avg'] or 0

    # Monthly scan counts (last 12 months)
    twelve_ago = timezone.now() - timedelta(days=365)
    monthly    = (records.filter(created_at__gte=twelve_ago)
                         .annotate(month=TruncMonth('created_at'))
                         .values('month')
                         .annotate(count=Count('id'))
                         .order_by('month'))

    # Condition breakdown
    conditions = (records.values('predicted_class', 'predicted_label')
                         .annotate(count=Count('id'))
                         .order_by('-count'))

    # Severity breakdown
    severities = (records.values('severity')
                         .annotate(count=Count('id'))
                         .order_by('severity'))

    # Confidence over time (last 20 scans)
    recent = records.order_by('-created_at')[:20].values(
        'created_at', 'confidence', 'predicted_class', 'severity')

    # Scans this week vs last week
    now       = timezone.now()
    week_ago  = now - timedelta(days=7)
    two_ago   = now - timedelta(days=14)
    this_week = records.filter(created_at__gte=week_ago).count()
    last_week = records.filter(created_at__gte=two_ago, created_at__lt=week_ago).count()
    week_change = this_week - last_week

    # Most common condition
    top_condition = conditions.first()

    # JSON-safe data for Chart.js
    monthly_labels = [m['month'].strftime('%b %Y') for m in monthly]
    monthly_counts = [m['count'] for m in monthly]
    cond_labels    = [c['predicted_label'] for c in conditions]
    cond_counts    = [c['count'] for c in conditions]
    sev_labels     = [s['severity'] for s in severities]
    sev_counts     = [s['count'] for s in severities]
    conf_labels    = [r['created_at'].strftime('%d %b') for r in reversed(list(recent))]
    conf_values    = [round(r['confidence'], 1) for r in reversed(list(recent))]

    return render(request, 'App/analytics.html', {
        'total':          total,
        'avg_conf':       round(avg_conf, 1),
        'this_week':      this_week,
        'week_change':    week_change,
        'top_condition':  top_condition,
        'is_doctor':      is_doctor,
        'monthly_labels': json.dumps(monthly_labels),
        'monthly_counts': json.dumps(monthly_counts),
        'cond_labels':    json.dumps(cond_labels),
        'cond_counts':    json.dumps(cond_counts),
        'sev_labels':     json.dumps(sev_labels),
        'sev_counts':     json.dumps(sev_counts),
        'conf_labels':    json.dumps(conf_labels),
        'conf_values':    json.dumps(conf_values),
    })


# ═══════════════════════════════════════════════
# FEATURE 2 — Doctor–Patient Assignment
# ═══════════════════════════════════════════════

@login_required
def my_patients(request):
    """Doctor views their assigned patients."""
    if not (hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'):
        raise PermissionDenied
    links = DoctorPatientLink.objects.filter(
        doctor=request.user, is_active=True
    ).select_related('patient', 'patient__profile')
    return render(request, 'App/my_patients.html', {'links': links})


@login_required
def assign_patient(request):
    """Doctor assigns a patient by username or patient ID."""
    if not (hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'):
        raise PermissionDenied
    if request.method == 'POST':
        identifier = request.POST.get('identifier', '').strip()
        notes      = request.POST.get('notes', '').strip()
        try:
            patient = User.objects.get(
                Q(username=identifier) | Q(email=identifier),
                profile__user_type='patient'
            )
            link, created = DoctorPatientLink.objects.get_or_create(
                doctor=request.user, patient=patient,
                defaults={'notes': notes}
            )
            if created:
                messages.success(request, f'Patient {patient.get_full_name() or patient.username} assigned successfully.')
                # Notify patient by email
                send_mail(
                    subject='Doctor Assigned — PneumoAI',
                    message=(f'Hello {patient.get_full_name() or patient.username},\n\n'
                             f'Dr. {request.user.get_full_name()} has been assigned to review your scans on PneumoAI.\n\n— PneumoAI'),
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[patient.email], fail_silently=True,
                )
            else:
                link.is_active = True; link.save()
                messages.info(request, 'Patient was already assigned and is now active.')
        except User.DoesNotExist:
            messages.error(request, 'No patient found with that username or email.')
    return redirect('my-patients')


@login_required
def unassign_patient(request, pk):
    """Doctor removes a patient assignment."""
    link = get_object_or_404(DoctorPatientLink, pk=pk, doctor=request.user)
    link.is_active = False; link.save()
    messages.success(request, 'Patient unassigned.')
    return redirect('my-patients')


@login_required
def patient_scans(request, patient_pk):
    """Doctor views a specific assigned patient's scan history."""
    patient = get_object_or_404(User, pk=patient_pk)
    is_linked = DoctorPatientLink.objects.filter(
        doctor=request.user, patient=patient, is_active=True).exists()
    if not is_linked:
        raise PermissionDenied
    records = ScanRecord.objects.filter(user=patient)
    return render(request, 'App/history.html', {
        'records': records, 'is_doctor': True,
        'viewing_patient': patient,
    })


# ═══════════════════════════════════════════════
# FEATURE 3 — Second Opinion Request
# ═══════════════════════════════════════════════

@login_required
def request_second_opinion(request, scan_pk):
    """Patient requests a second opinion from a doctor."""
    record  = get_object_or_404(ScanRecord, pk=scan_pk, user=request.user)
    doctors = User.objects.filter(profile__user_type='doctor').select_related('profile', 'doctorprofile')

    if request.method == 'POST':
        doctor_id = request.POST.get('doctor_id')
        message   = request.POST.get('message', '')
        urgency   = request.POST.get('urgency', 'normal')
        try:
            doctor = User.objects.get(pk=doctor_id, profile__user_type='doctor')
            sor, created = SecondOpinionRequest.objects.get_or_create(
                scan=record, patient=request.user, doctor=doctor,
                defaults={'message': message, 'urgency': urgency}
            )
            if created:
                # Email doctor
                send_mail(
                    subject=f'Second Opinion Request — PneumoAI [{urgency.upper()}]',
                    message=(f'Dear Dr. {doctor.get_full_name()},\n\n'
                             f'{request.user.get_full_name() or request.user.username} has requested a '
                             f'second opinion on their {record.predicted_label} scan.\n\n'
                             f'Urgency: {urgency}\nMessage: {message or "No message"}\n\n'
                             f'Please log in to PneumoAI to review.\n\n— PneumoAI'),
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[doctor.email], fail_silently=True,
                )
                messages.success(request, f'Second opinion request sent to Dr. {doctor.get_full_name()}.')
            else:
                messages.warning(request, 'You already have a pending request to this doctor.')
        except User.DoesNotExist:
            messages.error(request, 'Doctor not found.')
        return redirect('scan-detail', pk=scan_pk)

    return render(request, 'App/second_opinion_request.html', {
        'record': record, 'doctors': doctors,
    })


@login_required
def my_opinion_requests(request):
    """Doctor sees all second opinion requests assigned to them."""
    if not (hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'):
        raise PermissionDenied
    pending  = SecondOpinionRequest.objects.filter(doctor=request.user, status='pending')
    accepted = SecondOpinionRequest.objects.filter(doctor=request.user, status='accepted')
    reviewed = SecondOpinionRequest.objects.filter(doctor=request.user, status='reviewed')
    return render(request, 'App/opinion_requests.html', {
        'pending': pending, 'accepted': accepted, 'reviewed': reviewed,
    })


@login_required
def review_opinion(request, req_pk):
    """Doctor submits their review note."""
    sor = get_object_or_404(SecondOpinionRequest, pk=req_pk, doctor=request.user)
    if request.method == 'POST':
        note   = request.POST.get('doctor_note', '').strip()
        action = request.POST.get('action', 'review')
        if action == 'decline':
            sor.status = 'declined'; sor.save()
            messages.info(request, 'Request declined.')
        elif note:
            sor.mark_reviewed(note)
            # Email patient
            send_mail(
                subject='Second Opinion Ready — PneumoAI',
                message=(f'Dear {sor.patient.get_full_name() or sor.patient.username},\n\n'
                         f'Dr. {request.user.get_full_name()} has reviewed your {sor.scan.predicted_label} scan.\n\n'
                         f'Review Note:\n{note}\n\n'
                         f'Log in to PneumoAI to view the full report.\n\n— PneumoAI'),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[sor.patient.email], fail_silently=True,
            )
            messages.success(request, 'Review submitted and patient notified.')
        else:
            messages.error(request, 'Please enter a review note.')
    return redirect('opinion-requests')


# ═══════════════════════════════════════════════
# FEATURE 7 — Comparison View
# ═══════════════════════════════════════════════

@login_required
def compare_scans(request):
    """Side-by-side comparison of two scan records."""
    pk1 = request.GET.get('scan1')
    pk2 = request.GET.get('scan2')

    if not pk1 or not pk2:
        # Show scan picker
        records = ScanRecord.objects.filter(user=request.user).order_by('-created_at')
        return render(request, 'App/compare_pick.html', {'records': records})

    scan1 = get_object_or_404(ScanRecord, pk=pk1)
    scan2 = get_object_or_404(ScanRecord, pk=pk2)

    # Security: both must belong to user (or linked patient for doctor)
    for scan in [scan1, scan2]:
        is_linked = DoctorPatientLink.objects.filter(
            doctor=request.user, patient=scan.user, is_active=True).exists()
        if scan.user != request.user and not is_linked:
            raise PermissionDenied

    conf_change   = scan2.confidence - scan1.confidence
    same_result   = scan1.predicted_class == scan2.predicted_class
    improved      = (scan2.predicted_class == 'normal' and scan1.predicted_class != 'normal')
    worsened      = (scan1.predicted_class == 'normal' and scan2.predicted_class != 'normal')

    sev_order = {'None': 0, 'Low': 1, 'Moderate': 2, 'Severe': 3}
    sev1 = sev_order.get(getattr(scan1, 'severity', 'None'), 0)
    sev2 = sev_order.get(getattr(scan2, 'severity', 'None'), 0)
    severity_change = sev2 - sev1   # negative = improved

    s1_url = ''
    s2_url = ''
    try:
        if scan1.heatmap_image and scan1.heatmap_image.storage.exists(scan1.heatmap_image.name):
            s1_url = scan1.heatmap_image.url
    except (ValueError, OSError):
        pass
    try:
        if scan2.heatmap_image and scan2.heatmap_image.storage.exists(scan2.heatmap_image.name):
            s2_url = scan2.heatmap_image.url
    except (ValueError, OSError):
        pass

    return render(request, 'App/compare.html', {
        'scan1': scan1, 'scan2': scan2,
        'scan1_heatmap': s1_url, 'scan2_heatmap': s2_url,
        'conf_change': round(conf_change, 1),
        'same_result': same_result, 'improved': improved, 'worsened': worsened,
        'severity_change': severity_change,
    })


# ═══════════════════════════════════════════════
# FEATURE 8 — Admin Dashboard
# ═══════════════════════════════════════════════

@login_required
def admin_dashboard(request):
    """System-wide stats for staff/admin users."""
    if not request.user.is_staff:
        raise PermissionDenied

    now      = timezone.now()
    week_ago = now - timedelta(days=7)
    month_ago= now - timedelta(days=30)

    total_users    = User.objects.count()
    total_doctors  = Profile.objects.filter(user_type='doctor').count()
    total_patients = Profile.objects.filter(user_type='patient').count()
    total_scans    = ScanRecord.objects.count()
    scans_today    = ScanRecord.objects.filter(created_at__date=date.today()).count()
    scans_week     = ScanRecord.objects.filter(created_at__gte=week_ago).count()
    avg_confidence = ScanRecord.objects.aggregate(Avg('confidence'))['confidence__avg'] or 0

    condition_breakdown = (ScanRecord.objects
                           .values('predicted_class', 'predicted_label')
                           .annotate(count=Count('id'))
                           .order_by('-count'))

    severity_breakdown  = (ScanRecord.objects
                           .values('severity')
                           .annotate(count=Count('id'))
                           .order_by('-count'))

    top_doctors = (User.objects
                   .filter(profile__user_type='doctor')
                   .annotate(scan_count=Count('scan_records'))
                   .order_by('-scan_count')[:5])

    recent_scans = (ScanRecord.objects
                    .select_related('user')
                    .order_by('-created_at')[:15])

    # Monthly trend (last 6 months)
    six_ago = now - timedelta(days=180)
    monthly = (ScanRecord.objects
               .filter(created_at__gte=six_ago)
               .annotate(month=TruncMonth('created_at'))
               .values('month')
               .annotate(count=Count('id'))
               .order_by('month'))

    monthly_labels = json.dumps([m['month'].strftime('%b %Y') for m in monthly])
    monthly_counts = json.dumps([m['count'] for m in monthly])
    cond_labels    = json.dumps([c['predicted_label'] for c in condition_breakdown])
    cond_counts    = json.dumps([c['count'] for c in condition_breakdown])

    opinion_pending = SecondOpinionRequest.objects.filter(status='pending').count()

    return render(request, 'App/admin_dashboard.html', {
        'total_users': total_users, 'total_doctors': total_doctors,
        'total_patients': total_patients, 'total_scans': total_scans,
        'scans_today': scans_today, 'scans_week': scans_week,
        'avg_confidence': round(avg_confidence, 1),
        'condition_breakdown': condition_breakdown,
        'severity_breakdown': severity_breakdown,
        'top_doctors': top_doctors, 'recent_scans': recent_scans,
        'opinion_pending': opinion_pending,
        'monthly_labels': monthly_labels, 'monthly_counts': monthly_counts,
        'cond_labels': cond_labels, 'cond_counts': cond_counts,
    })


# ═══════════════════════════════════════════════
# BONUS — Patient Vitals Tracker
# ═══════════════════════════════════════════════

@login_required
def vitals(request):
    """Patient logs and views their vitals."""
    is_doctor = hasattr(request.user, 'profile') and request.user.profile.user_type == 'doctor'
    if is_doctor:
        raise PermissionDenied  # doctors view patient vitals via patient_scans

    if request.method == 'POST':
        form = VitalsForm(request.POST)
        if form.is_valid():
            v = form.save(commit=False)
            v.patient = request.user
            v.save()
            messages.success(request, 'Vitals recorded successfully.')
            return redirect('vitals')
    else:
        form = VitalsForm()

    history = PatientVitals.objects.filter(patient=request.user)[:20]

    # Chart data
    temp_data  = json.dumps([float(v.temperature) if v.temperature else None for v in reversed(list(history))])
    spo2_data  = json.dumps([float(v.spo2) if v.spo2 else None for v in reversed(list(history))])
    hr_data    = json.dumps([v.heart_rate for v in reversed(list(history))])
    date_labels= json.dumps([v.recorded_at.strftime('%d %b %H:%M') for v in reversed(list(history))])

    return render(request, 'App/vitals.html', {
        'form': form, 'history': history,
        'temp_data': temp_data, 'spo2_data': spo2_data,
        'hr_data': hr_data, 'date_labels': date_labels,
    })


# ═══════════════════════════════════════════════
# BONUS — Secure Report Share Link
# ═══════════════════════════════════════════════

@login_required
def create_share_link(request, pk):
    """Generate a time-limited shareable link for a report."""
    record = get_object_or_404(ScanRecord, pk=pk, user=request.user)
    share  = ReportShare.objects.create(
        record     = record,
        created_by = request.user,
        expires_at = timezone.now() + timedelta(hours=24),
    )
    share_url = request.build_absolute_uri(f'/report/share/{share.share_token}/')
    messages.success(request, f'Secure link created (valid 24h): {share_url}')
    return redirect('scan-detail', pk=pk)


def view_shared_report(request, token):
    """Anyone with the token can view the report (no login needed)."""
    share = get_object_or_404(ReportShare, share_token=token)
    if share.is_expired:
        return render(request, 'App/share_expired.html', status=410)
    share.accessed    = True
    share.access_count += 1
    share.save()
    record = share.record
    au = ''
    try:
        if record.heatmap_image and record.heatmap_image.storage.exists(record.heatmap_image.name):
            au = record.heatmap_image.url
    except (ValueError, OSError):
        au = ''
    return render(request, 'App/output.html', {
        'record': record, 'obj': record,
        'predict': record.predicted_class, 'predicted': record.predicted_label,
        'confidence': record.confidence, 'severity': getattr(record, 'severity', 'None'),
        'affected_image': au, 'affected_text': record.affected_text,
        'treatment_recs': record.get_treatment_recs(),
        'is_shared_view': True,
    })


# ═══════════════════════════════════════════════
# EXISTING VIEWS
# ═══════════════════════════════════════════════

@login_required
def Database(request):
    # Exclude records with no image file to prevent ValueError
    model_list = UserImageModel.objects.exclude(image='').exclude(image__isnull=True)
    return render(request, 'App/Database.html', {'models': model_list})


def logout_view(request):
    auth_logout(request)
    return redirect('/')