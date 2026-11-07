"""
PneumoAI - Grad-CAM Heatmap Generator

Compatible with the nested MobileNetV2 architecture used by PneumoAI.
Uses genuine gradients from the predicted class.
Does not generate artificial or random hotspots.
"""

import os
import uuid

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont


CLASSES = [
    "COVID_19",
    "normal",
    "opacity",
    "Pneumonia_Bacterial",
    "Pneumonia_Viral",
]

LAST_CONV = "Conv_1"


def _find_layer_recursive(model, layer_name):
    """Find a layer inside nested Keras models."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer

        if isinstance(layer, keras.Model):
            found = _find_layer_recursive(layer, layer_name)
            if found is not None:
                return found

    return None


def find_last_conv_layer(model):
    """Find the final convolutional layer in the nested model."""
    target = _find_layer_recursive(model, LAST_CONV)

    if target is not None:
        return target

    candidates = []

    def inspect(current):
        for layer in current.layers:
            if isinstance(
                layer,
                (keras.layers.Conv2D, keras.layers.DepthwiseConv2D),
            ):
                candidates.append(layer)

            if isinstance(layer, keras.Model):
                inspect(layer)

    inspect(model)

    if not candidates:
        raise ValueError("No convolutional layer found.")

    return candidates[-1]


def _get_gradcam_models(model):
    """
    Return:
      - backbone_model: input -> Conv_1 activation
      - conv_to_output_model: Conv_1 activation -> backbone output
      - feature_extractor_tail: layers after the nested backbone
      - classifier: classification head
    """

    if not isinstance(model, keras.Model):
        raise TypeError("The loaded model must be a Keras model.")

    if len(model.layers) < 2:
        raise ValueError("Unexpected outer model architecture.")

    feature_extractor = model.layers[0]
    classifier = model.layers[1]

    if not isinstance(feature_extractor, keras.Model):
        raise ValueError("Feature extractor is not a Keras model.")

    nested_model = None

    for layer in feature_extractor.layers:
        if isinstance(layer, keras.Model):
            if _find_layer_recursive(layer, LAST_CONV) is not None:
                nested_model = layer
                break

    if nested_model is None:
        raise ValueError("Nested MobileNetV2 model was not found.")

    conv_layer = _find_layer_recursive(nested_model, LAST_CONV)

    if conv_layer is None:
        raise ValueError(f"Could not find convolutional layer: {LAST_CONV}")

    try:
        backbone_model = keras.Model(
            inputs=nested_model.inputs,
            outputs=conv_layer.output,
            name="pneumoai_gradcam_backbone",
        )

        conv_to_output_model = keras.Model(
            inputs=conv_layer.output,
            outputs=nested_model.output,
            name="pneumoai_gradcam_tail",
        )

    except Exception as error:
        raise ValueError(
            f"Could not construct the Grad-CAM graph: {error}"
        ) from error

    nested_index = None

    for index, layer in enumerate(feature_extractor.layers):
        if layer is nested_model:
            nested_index = index
            break

    if nested_index is None:
        raise ValueError("Could not locate MobileNetV2 in feature extractor.")

    feature_extractor_tail = feature_extractor.layers[nested_index + 1:]

    return (
        backbone_model,
        conv_to_output_model,
        feature_extractor_tail,
        classifier,
    )


def generate_gradcam(model, image_array, class_index):
    """
    Generate a genuine Grad-CAM activation map.

    Returns a normalized 2D NumPy array.
    """

    class_index = int(class_index)

    if not 0 <= class_index < len(CLASSES):
        raise ValueError("Invalid class index.")

    (
        backbone_model,
        conv_to_output_model,
        feature_extractor_tail,
        classifier,
    ) = _get_gradcam_models(model)

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        activations = backbone_model(
            image_tensor,
            training=False,
        )

        tape.watch(activations)

        features = conv_to_output_model(
            activations,
            training=False,
        )

        for layer in feature_extractor_tail:
            try:
                features = layer(features, training=False)
            except TypeError:
                features = layer(features)

        predictions = classifier(
            features,
            training=False,
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        activations,
    )

    if gradients is None:
        raise ValueError(
            "Gradients are disconnected from Conv_1. "
            "Check the nested model architecture."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True,
    )

    cam = tf.reduce_sum(
        weights * activations,
        axis=-1,
    )

    cam = tf.nn.relu(cam)[0].numpy()

    if not np.all(np.isfinite(cam)):
        raise ValueError("Grad-CAM contains invalid values.")

    maximum = float(np.max(cam))

    if maximum <= 1e-8:
        raise ValueError(
            "No positive Grad-CAM activation was produced. "
            "No artificial heatmap was generated."
        )

    cam = cam / maximum

    return cam.astype(np.float32)


def resize_and_smooth(cam, width, height):
    """Resize and lightly smooth the genuine activation map."""

    resampling = getattr(Image, "Resampling", Image)

    cam_image = Image.fromarray(
        np.uint8(np.clip(cam, 0, 1) * 255),
        mode="L",
    )

    cam_image = cam_image.resize(
        (width, height),
        resampling.BICUBIC,
    )

    cam_image = cam_image.filter(
        ImageFilter.GaussianBlur(
            radius=max(1, min(width, height) * 0.003),
        )
    )

    result = np.asarray(cam_image).astype(np.float32) / 255.0

    maximum = float(result.max())

    if maximum <= 1e-8:
        raise ValueError("Resized activation map is empty.")

    return result / maximum


def jet_colormap(heat):
    """Convert activation values into a JET colour map."""

    heat = np.clip(heat, 0, 1)

    red = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)

    return np.stack(
        [red, green, blue],
        axis=-1,
    )


def blend_heatmap(original_image, cam):
    """Overlay genuine Grad-CAM activations on the original X-ray."""

    base = np.asarray(
        original_image.convert("RGB")
    ).astype(np.float32)

    colours = jet_colormap(cam)

    alpha = np.clip(
        (cam - 0.20) / 0.80,
        0,
        1,
    )

    alpha = np.power(alpha, 1.4) * 0.48
    alpha = alpha[..., np.newaxis]

    output = (
        base * (1 - alpha)
        + colours * 255 * alpha
    )

    return Image.fromarray(
        np.uint8(np.clip(output, 0, 255)),
        mode="RGB",
    )


def annotate_heatmap(image, pred_class, confidence):
    """Add a small identification header to the heatmap."""

    image = image.copy()
    draw = ImageDraw.Draw(image)

    width, height = image.size

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            max(10, height // 45),
        )
    except OSError:
        font = ImageFont.load_default()

    bar_height = max(24, height // 18)

    draw.rectangle(
        [(0, 0), (width, bar_height)],
        fill=(5, 12, 25),
    )

    label = (
        f"GRAD-CAM | {pred_class.replace('_', ' ')} "
        f"| Model score: {confidence:.1f}%"
    )

    draw.text(
        (8, 5),
        label,
        fill=(190, 225, 255),
        font=font,
    )

    return image


def generate_advanced_heatmap(
    image_pil,
    pred_class,
    confidence,
    severity,
    uid,
    media_root,
    media_url,
    model=None,
    boxes=None,
    class_index=None,
):
    """
    Generate a genuine Grad-CAM overlay.

    The model receives a 224x224 ImageOps.fit crop. The heatmap is
    overlaid on that same crop and pasted back into its matching
    position on the original image.
    """

    if model is None:
        print("[GradCAM] Model not supplied.")
        return ""

    if pred_class not in CLASSES:
        print("[GradCAM] Unknown prediction class.")
        return ""

    try:
        resampling = getattr(Image, "Resampling", Image)

        original = image_pil.convert("RGB")
        original_width, original_height = original.size

        if original_width < 1 or original_height < 1:
            raise ValueError("Invalid image dimensions.")

        # Same preprocessing used in views.py.
        model_image = ImageOps.fit(
            original,
            (224, 224),
            method=resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

        image_array = np.asarray(
            model_image,
            dtype=np.float32,
        )

        image_array = (image_array / 127.0) - 1.0
        image_array = np.expand_dims(image_array, axis=0)

        if class_index is None:
            class_index = CLASSES.index(pred_class)

        class_index = int(class_index)

        if class_index != CLASSES.index(pred_class):
            raise ValueError(
                "Class index does not match predicted class."
            )

        # Generate genuine Grad-CAM.
        raw_cam = generate_gradcam(
            model,
            image_array,
            class_index,
        )

        # Generate overlay in the model's 224x224 coordinates.
        cam = resize_and_smooth(raw_cam, 224, 224)

        cropped_overlay = blend_heatmap(
            model_image,
            cam,
        )

        # Find the central square crop in the original X-ray.
        if original_width > original_height:
            crop_width = float(original_height)
            crop_height = float(original_height)

            left = (original_width - crop_width) * 0.5
            top = 0.0

        else:
            crop_width = float(original_width)
            crop_height = float(original_width)

            left = 0.0
            top = (original_height - crop_height) * 0.5

        x1 = max(0, int(round(left)))
        y1 = max(0, int(round(top)))

        x2 = min(
            original_width,
            int(round(left + crop_width)),
        )

        y2 = min(
            original_height,
            int(round(top + crop_height)),
        )

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "Could not map Grad-CAM to original image."
            )

        # Resize overlay to the original crop dimensions.
        cropped_overlay = cropped_overlay.resize(
            (x2 - x1, y2 - y1),
            resampling.LANCZOS,
        )

        # Paste overlay at its corresponding original location.
        result = original.copy()

        result.paste(
            cropped_overlay,
            (x1, y1),
        )

        result = annotate_heatmap(
            result,
            pred_class,
            float(confidence),
        )

        # Save heatmap.
        filename = (
            f"gradcam_{pred_class}_"
            f"{uid or uuid.uuid4().hex[:8]}.png"
        )

        output_directory = os.path.join(
            str(media_root),
            "heatmap",
        )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        output_path = os.path.join(
            output_directory,
            filename,
        )

        result.save(
            output_path,
            format="PNG",
            optimize=True,
        )

        print(
            f"[GradCAM] Generated successfully: {filename}"
        )

        return (
            media_url.rstrip("/")
            + "/heatmap/"
            + filename
        )

    except Exception as error:
        print(
            f"[GradCAM] Heatmap generation failed: {error}"
        )
        return ""


def describe_from_boxes_enhanced(boxes, pred_class):
    """Return a cautious description of the model classification."""

    descriptions = {
        "COVID_19": (
            "The model classified this X-ray as COVID-19. "
            "Clinical review is required to assess the imaging findings."
        ),

        "Pneumonia_Bacterial": (
            "The model classified this X-ray as bacterial pneumonia. "
            "Clinical review is required to assess possible consolidation."
        ),

        "Pneumonia_Viral": (
            "The model classified this X-ray as viral pneumonia. "
            "Clinical review is required to assess the imaging findings."
        ),

        "opacity": (
            "The model classified this X-ray as lung opacity. "
            "Further radiological assessment is recommended."
        ),

        "normal": (
            "The model classified this X-ray as normal. "
            "A healthcare professional should interpret the image "
            "alongside clinical findings."
        ),
    }

    return descriptions.get(
        pred_class,
        "AI analysis completed. Please consult a qualified professional.",
    )


def generate_heatmap(
    image_pil,
    boxes,
    uid,
    pred_class="opacity",
    confidence=85.0,
    severity="Low",
    media_root=None,
    media_url="/media/",
    model=None,
    class_index=None,
):
    """Backward-compatible heatmap wrapper."""

    from django.conf import settings

    return generate_advanced_heatmap(
        image_pil=image_pil,
        pred_class=pred_class,
        confidence=confidence,
        severity=severity,
        uid=uid,
        media_root=media_root or settings.MEDIA_ROOT,
        media_url=media_url or getattr(
            settings,
            "MEDIA_URL",
            "/media/",
        ),
        model=model,
        boxes=boxes,
        class_index=class_index,
    )