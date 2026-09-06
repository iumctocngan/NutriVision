import base64
import binascii
import io
import json
import logging
import os
import re
import time
import unicodedata
import uuid

import boto3
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
MODEL_KEY = os.getenv("MODEL_KEY", "food_model.onnx")
CALORIE_MAP_KEY = os.getenv("CALORIE_MAP_KEY", "data/calorie_map.json")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
LOCAL_MODEL_PATH = "/tmp/food_model.onnx"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_PORTION_SCALES = {0.7, 1.0, 1.5, 2.0}

S3_CLIENT = boto3.client("s3", region_name=AWS_REGION)
REKOGNITION_CLIENT = boto3.client("rekognition", region_name=AWS_REGION)

_SESSION = None
_INPUT_NAME = None
_OUTPUT_NAME = None
_NUTRITION_DB = None
_CLASSES = None


class ConfigurationError(RuntimeError):
    pass


class RecognitionError(RuntimeError):
    pass


def normalize_text(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def get_bucket_name():
    bucket_name = os.getenv("S3_BUCKET_NAME", "fcaj-food-ai-storage-2026-sg-110359221458-ap-southeast-1-an")
    return bucket_name


def initialize_model():
    """Download artifacts once per execution environment and cache the ONNX session."""
    global _SESSION, _INPUT_NAME, _OUTPUT_NAME, _NUTRITION_DB, _CLASSES
    if _SESSION is not None:
        return _SESSION, _NUTRITION_DB, _CLASSES

    bucket_name = get_bucket_name()
    model_path = LOCAL_MODEL_PATH
    if not os.path.isfile(model_path):
        if os.path.isfile("food_model.onnx"):
            model_path = "food_model.onnx"
            logger.info("Using bundled food_model.onnx file.")
        else:
            logger.info("Downloading model from s3://%s/%s", bucket_name, MODEL_KEY)
            S3_CLIENT.download_file(bucket_name, MODEL_KEY, model_path)

    if os.path.isfile("calorie_map.json"):
        logger.info("Loading calorie_map.json from local bundle.")
        with open("calorie_map.json", "r", encoding="utf-8") as f:
            database = json.load(f)
    elif os.path.isfile("data/calorie_map.json"):
        logger.info("Loading data/calorie_map.json from local bundle.")
        with open("data/calorie_map.json", "r", encoding="utf-8") as f:
            database = json.load(f)
    else:
        logger.info("Fetching calorie map from s3://%s/%s", bucket_name, CALORIE_MAP_KEY)
        calorie_response = S3_CLIENT.get_object(
            Bucket=bucket_name,
            Key=CALORIE_MAP_KEY,
        )
        database = json.loads(calorie_response["Body"].read().decode("utf-8"))
    if not isinstance(database, dict) or not database:
        raise ConfigurationError("calorie_map.json must contain a non-empty object.")

    classes = sorted(database.keys())
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    output_shape = session.get_outputs()[0].shape
    output_classes = output_shape[-1] if output_shape else None
    if isinstance(output_classes, int) and output_classes != len(classes):
        raise ConfigurationError(
            f"Model outputs {output_classes} classes but calorie_map has {len(classes)}."
        )

    _SESSION = session
    _INPUT_NAME = session.get_inputs()[0].name
    _OUTPUT_NAME = session.get_outputs()[0].name
    _NUTRITION_DB = database
    _CLASSES = classes
    logger.info("ONNX model initialized with %d classes.", len(classes))
    return _SESSION, _NUTRITION_DB, _CLASSES


def decode_image(image_data):
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("Missing required field 'image'.")
    if image_data.startswith("data:"):
        if "," not in image_data:
            raise ValueError("Invalid image data URL.")
        image_data = image_data.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Image must be valid base64 data.") from error
    if not image_bytes:
        raise ValueError("Image data is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 5 MB limit.")
    try:
        source_image = Image.open(io.BytesIO(image_bytes))
        if source_image.format not in ("JPEG", "PNG"):
            raise ValueError("Unsupported image format; use JPEG or PNG.")
        image = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Image is corrupted or unsupported; use JPEG or PNG.") from error
    return image_bytes, image


def check_image_quality(image):
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(np.mean(grayscale))
    variance = float(np.var(grayscale))
    if brightness < 20:
        return False, "Ảnh quá tối. Vui lòng chụp lại ở nơi đủ ánh sáng."
    if brightness > 245:
        return False, "Ảnh bị chói sáng quá mức."
    if variance < 80:
        return False, "Ảnh thiếu chi tiết hoặc có thể quá mờ."
    return True, None


def preprocess_image(image):
    resized = image.resize((288, 288), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0).astype(np.float32)


def run_onnx_inference(tensor):
    session, database, classes = initialize_model()
    logits = session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})[0][0]
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
    predicted_index = int(np.argmax(probabilities))
    if predicted_index >= len(classes):
        raise ConfigurationError("Model output index is outside the class list.")
    return classes[predicted_index], float(probabilities[predicted_index]), database


def find_rekognition_match(labels, database):
    normalized_keys = {normalize_text(key): key for key in database}
    normalized_names = {
        normalize_text(info.get("name", "")): key
        for key, info in database.items()
        if info.get("name")
    }
    # Aliases xử lý 2 trường hợp:
    # 1. Rekognition trả nhãn số ít nhưng key trong calorie_map là số nhiều (donut vs donuts)
    # 2. Rekognition trả nhãn chung chung (noodle, soup, cake) → map sang món gần nhất
    aliases = {
        # --- BUG FIX: salad không phải key hợp lệ, ramen có key riêng ---
        "salad": "caesar_salad",
        "ramen": "ramen",

        # --- SỐ ÍT / SỐ NHIỀU ---
        "donut": "donuts",
        "waffle": "waffles",
        "pancake": "pancakes",
        "taco": "tacos",
        "dumpling": "dumplings",
        "spring_roll": "spring_rolls",
        "cupcake": "cup_cakes",
        "macaron": "macarons",
        "churro": "churros",
        "beignet": "beignets",

        # --- BIẾN THỂ CHÍNH TẢ ---
        "lasagne": "lasagna",
        "hotdog": "hot_dog",
        "macaroni": "macaroni_and_cheese",

        # --- NHÃN CHUNG CHUNG → MÓN GẦN NHẤT ---
        # Fast food & bánh mì
        "burger": "hamburger",
        "hamburger": "hamburger",
        "sandwich": "club_sandwich",
        "pizza": "pizza",

        # Thịt nướng
        "steak": "steak",
        "beef": "steak",
        "filet": "filet_mignon",
        "rib": "baby_back_ribs",
        "bbq": "baby_back_ribs",

        # Gà
        "wing": "chicken_wings",
        "curry": "chicken_curry",

        # Mì / Súp / Cơm
        "noodle": "pho",
        "soup": "pho",
        "pasta": "spaghetti_bolognese",
        "spaghetti": "spaghetti_bolognese",
        "rice": "fried_rice",
        "pad_thai": "pad_thai",

        # Hải sản / Nhật / Á
        "sushi": "sushi",
        "gyoza": "gyoza",
        "spring": "spring_rolls",
        "dumpling": "dumplings",
        "falafel": "falafel",
        "samosa": "samosa",
        "bibimbap": "bibimbap",

        # Salad
        "greek": "greek_salad",
        "caprese": "caprese_salad",
        "caesar": "caesar_salad",

        # Khoai tây / Snack
        "fries": "french_fries",
        "chip": "nachos",
        "nacho": "nachos",

        # Bánh ngọt / Tráng miệng
        "cake": "chocolate_cake",
        "chocolate": "chocolate_cake",
        "cheesecake": "cheesecake",
        "carrot": "carrot_cake",
        "pudding": "bread_pudding",
        "ice_cream": "ice_cream",
        "baklava": "baklava",

        # Điểm tâm
        "egg": "omelette",
        "benedict": "eggs_benedict",
        "pancake": "pancakes",
        "waffle": "waffles",

        # Bánh mì
        "bread": "garlic_bread",
        "garlic": "garlic_bread",
    }
    for label in labels:
        label_key = normalize_text(label.get("Name", ""))
        confidence = round(float(label.get("Confidence", 0)) / 100.0, 4)
        if label_key in normalized_keys:
            return normalized_keys[label_key], confidence
        if label_key in normalized_names:
            return normalized_names[label_key], confidence
        for keyword, target_key in aliases.items():
            if keyword in label_key and target_key in database:
                return target_key, confidence
    return None



def run_rekognition_fallback(image_bytes, database):
    try:
        response = REKOGNITION_CLIENT.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=15,
            MinConfidence=40.0,
        )
    except Exception as error:
        logger.exception("Rekognition fallback failed.")
        raise RecognitionError("Fallback recognition is temporarily unavailable.") from error
    labels = response.get("Labels", [])
    logger.info(
        "Fallback labels: %s",
        [(label.get("Name"), round(label.get("Confidence", 0), 1)) for label in labels],
    )
    return find_rekognition_match(labels, database)


def save_ood_image(image_bytes, confidence):
    object_key = (
        f"ood_logs/{time.strftime('%Y/%m/%d')}/"
        f"{uuid.uuid4().hex}_{confidence:.4f}.jpg"
    )
    try:
        S3_CLIENT.put_object(
            Bucket=get_bucket_name(),
            Key=object_key,
            Body=image_bytes,
            ContentType="image/jpeg",
            Metadata={"onnx-confidence": f"{confidence:.4f}"},
        )
        return object_key
    except Exception:
        logger.exception("Unable to store OOD image; continuing without the audit image.")
        return None


def make_response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
            "Content-Type": "application/json; charset=utf-8",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


def parse_body(event):
    body = event.get("body")
    if body is None or body == "":
        raise ValueError("Missing request body.")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON.") from error
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    return body


def parse_portion_scale(value):
    try:
        portion_scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("portion_scale must be a number.") from error
    if portion_scale not in ALLOWED_PORTION_SCALES:
        raise ValueError("portion_scale must be one of: 0.7, 1.0, 1.5, 2.0.")
    return portion_scale


def lambda_handler(event, context):
    start_time = time.perf_counter()
    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or ""
    ).upper()
    if method == "OPTIONS":
        return make_response(200, {"message": "CORS OK"})

    try:
        body = parse_body(event)
        image_bytes, image = decode_image(body.get("image"))
        portion_scale = parse_portion_scale(body.get("portion_scale", 1.0))
        quality_ok, quality_message = check_image_quality(image)
        if not quality_ok:
            return make_response(422, {
                "status": "warning",
                "warning_code": "POOR_IMAGE_QUALITY",
                "message": quality_message,
            })

        food_key, confidence, database = run_onnx_inference(preprocess_image(image))
        engine = "EfficientNet-B0 ONNX"
        ood_key = None
        if confidence < CONFIDENCE_THRESHOLD:
            ood_key = save_ood_image(image_bytes, confidence)
            fallback = run_rekognition_fallback(image_bytes, database)
            if fallback is None:
                return make_response(422, {
                    "status": "error",
                    "error_code": "FOOD_NOT_RECOGNIZED",
                    "message": "Không nhận diện được món ăn trong ảnh. Vui lòng thử ảnh khác.",
                    "onnx_confidence": round(confidence, 4),
                })
            food_key, confidence = fallback
            engine = "Amazon Rekognition fallback"

        info = database[food_key]
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return make_response(200, {
            "status": "success",
            "food_class": food_key,
            "name": info["name"],
            "category": info.get("category", "Món ăn"),
            "confidence": round(confidence, 4),
            "portion_scale": portion_scale,
            "estimated_calories": int(round(float(info["calories"]) * portion_scale)),
            "macronutrients": {
                "protein_g": round(float(info.get("protein_g", 0)) * portion_scale, 1),
                "carbs_g": round(float(info.get("carbs_g", 0)) * portion_scale, 1),
                "fat_g": round(float(info.get("fat_g", 0)) * portion_scale, 1),
                "fiber_g": round(float(info.get("fiber_g", 0)) * portion_scale, 1),
            },
            "metadata": {
                "inference_engine": engine,
                "onnx_threshold": CONFIDENCE_THRESHOLD,
                "ood_log_key": ood_key,
                "aws_region": AWS_REGION,
                "latency_ms": elapsed_ms,
                "request_id": getattr(context, "aws_request_id", "local-test"),
            },
        })
    except ValueError as error:
        return make_response(400, {
            "status": "error", "error_code": "INVALID_REQUEST", "message": str(error),
        })
    except ConfigurationError:
        logger.exception("Lambda configuration error.")
        return make_response(500, {
            "status": "error", "error_code": "CONFIGURATION_ERROR",
            "message": "Server configuration is incomplete.",
        })
    except RecognitionError as error:
        return make_response(502, {
            "status": "error", "error_code": "RECOGNITION_ERROR", "message": str(error),
        })
    except Exception:
        logger.exception("Unexpected Lambda error.")
        return make_response(500, {
            "status": "error", "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected server error occurred.",
        })
