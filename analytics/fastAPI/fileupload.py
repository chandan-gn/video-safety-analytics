from fastapi import FastAPI, UploadFile                                                                                                     
import shutil                                                                                                                               
from pathlib import Path                                                                                                                    
                                                                                                                                            
app = FastAPI()                                                                                                                             
                
UPLOAD_DIR = Path("uploads")                                                                                                                
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:                                                                                                              
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "saved_to": str(dest)}  
