import os
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image
from torchvision import transforms

model_name = os.getenv("MODEL_NAME", "hair_classifier_empty.onnx")

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = preprocess(img).unsqueeze(0).numpy()
    return img_tensor

session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict(url):
    image = download_image(url)
    resized_img = prepare_image(image, [200,200])
    X = preprocess_image(resized_img)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    return float_predictions

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
