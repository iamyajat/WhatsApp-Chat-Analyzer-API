from re import split
from fastapi import FastAPI, File, HTTPException, UploadFile
from functions import txt_to_json as ttjson
import numpy as np

app = FastAPI(
    title="WhatsApp Analyzer",
    version="1.0",
    description="Get beautiful insights about your chats!",
)


@app.get("/")
async def root():
    return {"message": "Service is up!"}


@app.post("/txt_to_json/")
async def txt_to_json(file: UploadFile = File(...)):
    """Upload WhatsApp chats as .txt"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    x = contents.decode("utf-8")
    arr = split("\n", x)
    y = ttjson.convert(arr)
    return {"no_of_messages": len(y), "messages": y}
