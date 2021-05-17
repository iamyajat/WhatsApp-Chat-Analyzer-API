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


@app.post("/chats_to_json/")
async def chats_to_json(file: UploadFile = File(...)):
    """Get your chats in JSON format. (Upload WhatsApp chats as .txt)"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    resp = wa.chats_to_json(chats)
    return resp


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    """Get an analysis of your chats. (Upload WhatsApp chats as .txt)"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    resp = wa.analyze(chats)
    return resp


@app.post("/random/")
async def random(n: int = 10, file: UploadFile = File(...)):
    """Get random n chats. (Upload WhatsApp chats as .txt)"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    resp = wa.random_chats(chats, n)
    return resp
