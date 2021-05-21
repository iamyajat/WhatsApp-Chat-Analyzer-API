from re import split
from fastapi import FastAPI, File, HTTPException, UploadFile
import whatsapp_analyzer as wa
from starlette.responses import StreamingResponse
import io
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="WhatsApp Analyzer",
    version="1.0",
    description="Get beautiful insights about your chats!",
)

origins = [
    "http://wa-chat-analyzer.herokuapp.com/",
    "https://wa-chat-analyzer.herokuapp.com/",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:0000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/word_cloud/")
async def word_cloud(file: UploadFile = File(...)):
    """Get a word cloud"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    img = wa.word_cloud(chats)
    buf = io.BytesIO()
    plt.imsave(buf, img, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": 'inline; filename="%s.jpg"' % (file.filename[:-4],)
        },
    )
