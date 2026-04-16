from fastapi import APIRouter, UploadFile, File, HTTPException
from src.api.v1.schemas.query_schema import QueryRequest
from src.api.v1.services.query_service import query_documents 
import tempfile
from src.ingestion.ingestion import run_ingestion
import os

router = APIRouter()

@router.post("/query")
def query_endpoint(request: QueryRequest):
    results= query_documents(request.query)
    return results

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSION = {".pdf"}

@router.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    try:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_EXTENSION:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_path = os.path.join(UPLOAD_DIR, filename)


        with open(file_path, "wb") as f:
            f.write(await file.read())

        result = run_ingestion(file_path)

        return {
            "status": "success",
            "file_name": filename,
            "details": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
