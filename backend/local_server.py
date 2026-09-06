"""
Localhost Edge Case Testing Server (Python + ONNX Runtime + BaseHTTPRequestHandler)
Chạy thử nghiệm toàn bộ logic xử lý trường hợp biên (Edge Cases) trực tiếp trên máy cục bộ
Endpoint: http://127.0.0.1:5000/predict
"""

import os
import sys
import json
import base64
import time
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort
from http.server import HTTPServer, BaseHTTPRequestHandler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global Loaded Model & Data
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "food_model.onnx")
CALORIE_MAP_PATH = os.path.join(os.path.dirname(__file__), "calorie_map.json")

print(f"📦 Đang nạp mô hình ONNX: {ONNX_MODEL_PATH}")
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"🥗 Đang nạp bảng dữ liệu calo: {CALORIE_MAP_PATH}")
with open(CALORIE_MAP_PATH, "r", encoding="utf-8") as f:
    CALORIE_DB = json.load(f)

# DYNAMIC 50-CLASS ALPHABETICAL ORDER MATCHING PYTORCH TRAINED MODEL
CLASSES = sorted(list(CALORIE_DB.keys()))
print(f"✅ Đã nạp thành công {len(CLASSES)} lớp món ăn vào bộ nhớ!")

CONFIDENCE_THRESHOLD = 0.60  # Edge Case Cutoff (60%)

def preprocess_image(pil_img, target_size=(288, 288)):
    """Preprocess PIL Image to Normalized ONNX Tensor (1, 3, 288, 288)"""
    img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_data = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    # ImageNet Mean & Std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Change shape from (288, 288, 3) to (1, 3, 288, 288)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

def check_image_quality(pil_img):
    """EDGE CASE CHECK: Image Blur & Darkness Check"""
    img_gray = pil_img.convert('L')
    arr = np.array(img_gray)
    
    mean_brightness = np.mean(arr)
    variance = np.var(arr)
    
    if mean_brightness < 20:
        return False, "Ảnh quá tối. Vui lòng chụp lại ở nơi đủ ánh sáng."
    if mean_brightness > 245:
        return False, "Ảnh bị chói sáng quá mức."
    if variance < 80:
        return False, "Ảnh quá mờ hoặc bị mất chi tiết nét."
    
    return True, "OK"

def process_predict_request(body_json):
    start_time = time.time()
    
    # EDGE CASE 1: Missing image field
    if not body_json or 'image' not in body_json:
        return 400, {
            "status": "error",
            "error_code": "MISSING_IMAGE",
            "message": "Thiếu tham số 'image' (dạng chuỗi mã hóa base64)."
        }
    
    img_b64 = body_json['image']
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]
        
    # EDGE CASE 2: Corrupted Base64 Image
    try:
        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        return 400, {
            "status": "error",
            "error_code": "CORRUPTED_IMAGE",
            "message": f"Tệp hình ảnh bị hỏng hoặc không đúng định dạng JPG/PNG ({str(e)})."
        }

    # EDGE CASE 3: Image Quality Check (Blur / Dark)
    is_good_quality, quality_msg = check_image_quality(pil_img)
    if not is_good_quality:
        return 422, {
            "status": "warning",
            "warning_code": "POOR_IMAGE_QUALITY",
            "message": quality_msg
        }

    # Run ONNX Model Inference
    tensor_input = preprocess_image(pil_img)
    outputs = session.run([output_name], {input_name: tensor_input})[0]
    
    # Softmax probabilities
    exp_logits = np.exp(outputs[0] - np.max(outputs[0]))
    probs = exp_logits / np.sum(exp_logits)
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    
    if pred_idx < len(CLASSES):
        pred_class = CLASSES[pred_idx]
    else:
        pred_class = "pho"
    
    portion_scale = float(body_json.get('portion_scale', 1.0)) # Edge Case 4: Portion Multiplier

    # EDGE CASE 5: Out-of-Distribution / Low Confidence Cutoff (< 60%)
    if confidence < CONFIDENCE_THRESHOLD:
        return 200, {
            "status": "low_confidence_warning",
            "food_class": pred_class,
            "confidence": round(confidence, 4),
            "warning_message": f"Món ăn này có độ tin cậy thấp ({confidence*100:.1f}% < 60%). Có thể nằm ngoài danh mục 50 món hỗ trợ.",
            "estimated_calories": None,
            "macronutrients": None
        }

    # Retrieve Calorie Info
    info = CALORIE_DB.get(pred_class, CALORIE_DB.get('pho', list(CALORIE_DB.values())[0]))
    base_calories = info['calories']
    scaled_calories = int(round(base_calories * portion_scale))

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return 200, {
        "status": "success",
        "food_class": pred_class,
        "name": info['name'],
        "category": info.get('category', 'Món ăn'),
        "confidence": round(confidence, 4),
        "portion_scale": portion_scale,
        "estimated_calories": scaled_calories,
        "macronutrients": {
            "protein_g": round(info.get('protein_g', 22.0) * portion_scale, 1),
            "carbs_g": round(info.get('carbs_g', 54.0) * portion_scale, 1),
            "fat_g": round(info.get('fat_g', 8.5) * portion_scale, 1),
            "fiber_g": round(info.get('fiber_g', 3.2) * portion_scale, 1)
        },
        "metadata": {
            "inference_engine": "EfficientNet-B0 Fine-Tuned ONNX Model",
            "latency_ms": elapsed_ms,
            "threshold_applied": CONFIDENCE_THRESHOLD
        }
    }

class LocalRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                body_json = json.loads(post_data.decode('utf-8'))
            except Exception:
                body_json = {}

            status_code, response_data = process_predict_request(body_json)
            self._set_headers(status_code)
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

def run_server(port=5000):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, LocalRequestHandler)
    print(f"🚀 LOCAL EDGE CASE TEST SERVER ĐANG CHẠY TẠI: http://127.0.0.1:{port}/predict")
    print(f"  - Đã sẵn sàng nhận diện cho {len(CLASSES)} món ăn...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
