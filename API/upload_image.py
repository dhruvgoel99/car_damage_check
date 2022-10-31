from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File
from services.prediction import check_file, read_imagefile, generate_repair_report
# from services.prediction import predict

router = APIRouter(prefix='/scratch_detection', tags=["Images"])

@router.post('/multi_img')
async def root(files: List[UploadFile] = File(...)):
    count = 0
    report = dict()
    for file in files:
        if check_file(file) == "ok":
            image = read_imagefile(await file.read())
            prediction = generate_repair_report(image, file.filename)
            report[file.filename] = prediction
            count+=1
    if count == len(files):
        return report, 200
    else:
        raise HTTPException(
                status_code=500, detail="File doesn't match extension!"
            )