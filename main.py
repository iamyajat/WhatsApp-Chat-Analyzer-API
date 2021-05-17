from re import split
from fastapi import FastAPI, File, HTTPException, UploadFile
import whatsapp_analyzer as wa 
import numpy as np

app = FastAPI(
    title="WhatsApp Analyzer",
    version="1.0",
    description="Get beautiful insights about your chats!",
)


@app.get("/")
async def root():
    return {"message": "Service is up!"}


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    """Upload WhatsApp chats as .txt"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    resp = wa.analyze(chats)
    return resp
