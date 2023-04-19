from ray import serve
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict




@serve.deployment
class ImageModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    async def __call__(self, starlette_request: Request) -> Dict:
        image_payload_bytes = await starlette_request.body()
        pil_image = Image.open(BytesIO(image_payload_bytes))
        
        results = self.model(pil_image)
        
        return {"bbox":results[0].boxes.xyxy.tolist(),"class":results[0].boxes.cls.tolist()}

image_model = ImageModel.bind()