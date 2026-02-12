import os
import base64
import uuid
import numpy as np
from PIL import Image
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.http import HttpResponseForbidden
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth import get_user_model
from django.views.decorators.http import require_http_methods

from .models import ECGRecord
from .forms import ECGUploadForm

from openai import OpenAI

# ------------------- CONSTANTS -------------------
IMG_SIZE = (128, 128)
THRESHOLD = 0.5
BINARY_MODEL_PATH = os.path.join(os.getcwd(), 'ecgapp', 'ml_model', 'ecg_binary_model.h5')
# ------------------- LOAD MODEL (LAZY) -------------------
_binary_model = None

def get_binary_model():
    global _binary_model
    if _binary_model is None:
        _binary_model = load_model(BINARY_MODEL_PATH)
    return _binary_model

LABELS = {
    1: "✅ Normal ECG: No arrhythmia detected.",
    0: "⚠️ Abnormal ECG: Possible irregular heartbeat. Consult a cardiologist."
}

UserModel = get_user_model()

# ------------------- OPENAI CLIENT -------------------
client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_heart_image(is_normal: bool) -> str:
    """
    Generates a 3D-style heart image and returns RELATIVE media path
    """
    if is_normal:
        prompt = (
            "Highly detailed 3D medical illustration of a healthy human heart, "
            "realistic anatomy, normal electrical rhythm, medical textbook style, "
            "dark background, ultra high resolution"
        )
    else:
        prompt = (
            "Highly detailed 3D medical illustration of a human heart with arrhythmia, "
            "irregular glowing electrical signals in red, stressed ventricles, "
            "clinical medical visualization, dark background, ultra realistic"
        )

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    heart_dir = settings.MEDIA_ROOT / "heart_views"
    os.makedirs(heart_dir, exist_ok=True)

    filename = f"heart_{uuid.uuid4().hex}.png"
    filepath = heart_dir / filename

    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # IMPORTANT: return relative path
    return f"heart_views/{filename}"


# ------------------- HOME -------------------
def home(request):
    return render(request, 'home.html')

# ------------------- ADMIN LOGIN -------------------
def admin_login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user and user.is_staff:
            login(request, user)
            return redirect('admin_dashboard')
        else:
            return render(request, 'admin/admin_login.html', {'error': 'Invalid admin credentials'})
    return render(request, 'admin/admin_login.html')

# ------------------- USER LOGIN -------------------
def user_login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user and not user.is_staff:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'user/login.html', {'error': 'Invalid user credentials'})
    return render(request, 'user/login.html')

# ------------------- USER REGISTER -------------------
def user_register_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():
            return render(request, 'user/register.html', {'error': 'Username already exists'})

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, 'Account created successfully. Please log in.')
        return redirect('login')
    return render(request, 'user/register.html')

# ------------------- LOGOUT -------------------
def logout_view(request):
    logout(request)
    return redirect('home')

# ------------------- ADMIN DASHBOARD -------------------
@login_required
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('home')
    return render(request, 'admin/dashboard.html')

# ------------------- UPLOAD ECG + PREDICT -------------------
@login_required
def upload_ecg(request):
    if request.user.is_staff:
        return redirect('home')

    prediction = None
    error = None
    record = None

    if request.method == 'POST':
        form = ECGUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            filename = file.name.lower()

            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                error = "Only image files (.png, .jpg, .jpeg) are supported."
            else:
                try:
                    # Preprocess ECG image
                    img = Image.open(file).resize(IMG_SIZE).convert('RGB')
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    model = get_binary_model()
                    pred = float(model.predict(img_array)[0][0])

                    label = 1 if pred >= THRESHOLD else 0
                    prediction = LABELS[label]

                    # Generate heart visualization
                    heart_image_path = generate_heart_image(is_normal=(label == 1))

                    # Save record
                    record = ECGRecord.objects.create(
                        uploaded_by=request.user,
                        filename=file.name,
                        result=prediction,
                        confidence=pred,
                        file_type='image',
                        heart_image=heart_image_path
                    )

                except Exception as e:
                    error = f"Prediction error: {str(e)}"
    else:
        form = ECGUploadForm()

    return render(request, 'user/upload_ecg.html', {
        'form': form,
        'prediction': prediction,
        'error': error,
        'icon': 'success' if prediction and 'Normal' in prediction else 'warning',
        'record': record
    })

# ------------------- ADMIN VIEW ALL ECG RECORDS -------------------
@login_required
def view_all_ecg_records(request):
    if not request.user.is_staff:
        return redirect('home')

    records = ECGRecord.objects.all().order_by('-uploaded_at')
    return render(request, 'admin/view_ecg_records.html', {'records': records})

# ------------------- USER VIEW OWN ECG RECORDS -------------------
@login_required
def view_own_ecg_records(request):
    if request.user.is_staff:
        return redirect('admin_dashboard')

    records = ECGRecord.objects.filter(uploaded_by=request.user).order_by('-uploaded_at')
    return render(request, 'user/view_ecg_records.html', {'records': records})

# ------------------- TRAINING PLOT VIEW -------------------
@login_required
def view_training_plot(request):
    if not request.user.is_staff:
        return redirect('home')
    return render(request, 'metrics/training_plot.html')

# ------------------- DELETE ECG RECORD -------------------
@login_required
def delete_ecg_record(request, record_id):
    try:
        record = ECGRecord.objects.get(id=record_id)
    except ECGRecord.DoesNotExist:
        return redirect('home')
    # Only allow owner or admin to delete
    if request.user.is_staff or record.uploaded_by == request.user:
        record.delete()
        if request.user.is_staff:
            return redirect('view_ecg_records')
        else:
            return redirect('user_view_ecg_records')
    else:
        return HttpResponseForbidden("You do not have permission to delete this record.")

# ------------------- ADMIN USER MANAGEMENT -------------------
@login_required
def admin_user_list(request):
    if not request.user.is_staff:
        return redirect('home')
    users = UserModel.objects.all().order_by('-date_joined')
    return render(request, 'admin/user_list.html', {'users': users})

@login_required
@require_http_methods(["GET", "POST"])
def admin_user_edit(request, user_id):
    if not request.user.is_staff:
        return redirect('home')
    user = UserModel.objects.get(pk=user_id)
    if request.method == 'POST':
        form = UserChangeForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('admin_user_list')
    else:
        form = UserChangeForm(instance=user)
    return render(request, 'admin/user_edit.html', {'form': form, 'user_obj': user})

@login_required
@require_http_methods(["POST"])
def admin_user_delete(request, user_id):
    if not request.user.is_staff:
        return redirect('home')
    user = UserModel.objects.get(pk=user_id)
    if user != request.user:  # Prevent admin from deleting themselves
        user.delete()
    return redirect('admin_user_list')


