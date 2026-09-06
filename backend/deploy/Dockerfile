FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code and model artifacts into Lambda task root
COPY lambda_function.py food_model.onnx calorie_map.json ./

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
