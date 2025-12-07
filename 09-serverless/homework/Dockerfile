FROM agrigorev/model-2025-hairstyle:v1

RUN pip install onnxruntime urllib3 Pillow torchvision

#COPY hair_classifier_empty.onnx.data .
#COPY hair_classifier_empty.onnx .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]