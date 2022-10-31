from fastapi import FastAPI
from API.upload_image import router as r1

app = FastAPI()

app.include_router(r1)