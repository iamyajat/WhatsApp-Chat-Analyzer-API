from re import split
from fastapi import FastAPI, File, HTTPException, UploadFile
import src.whatsapp_analyzer as wa
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import PlainTextResponse
from starlette.responses import RedirectResponse


app = FastAPI(
    title="WhatsApp Analyzer",
    version="2.0",
    description="Get beautiful insights about your chats!",
)

print("DOCS:", "http://127.0.0.1:8000/docs")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    print(exc.detail)
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.get("/")
async def root():
    response = RedirectResponse(url="https://ourchatstory.co")
    return response


@app.post("/chats_to_json")
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


@app.post("/analyze")
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


@app.post("/throwback")
async def random(n: int = 10, file: UploadFile = File(...)):
    """Get a set of n old chats. (Upload WhatsApp chats as .txt)"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    resp = wa.throwback_chats(chats, n)
    return resp


@app.post("/wordcloud")
async def word_cloud(file: UploadFile = File(...)):
    """Get a word cloud"""
    extension = file.filename.split(".")[-1] in ("txt", "TXT")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload .txt files only!")
    contents = await file.read()
    decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    img = wa.get_word_cloud(chats)
    # buf = io.BytesIO()
    # plt.imsave(buf, img, format="PNG")
    # buf.seek(0)
    # return StreamingResponse(
    #     buf,
    #     media_type="image/jpeg",
    #     headers={
    #         "Content-Disposition": 'inline; filename="%s.jpg"' % (file.filename[:-4],)
    #     },
    # )
    return img


@app.post("/wrap")
async def wrap(file: UploadFile = File(...)):
    """WhatsApp Wrap 2022"""
    file_type = file.filename.split(".")[-1]
    extension = file_type in ("txt", "TXT", "zip", "ZIP")
    print("\n\n---------------------------------------------")
    print(" "+file.filename.split(".")[0])
    print("---------------------------------------------")
    if not extension:
        raise HTTPException(
            status_code=400, detail="Please upload .txt or .zip files only!"
        )
    contents = await file.read()
    decoded_contents = ""
    if file_type == "zip" or file_type == "ZIP":
        try:
            decoded_contents = wa.extract_zip(contents)["_chat.txt"].decode("utf-8")
        except:
            raise HTTPException(
                status_code=400, detail="Zip file is corrupted! Please try again."
            )
    else:
        decoded_contents = contents.decode("utf-8")
    chats = split("\n", decoded_contents)
    resp = wa.wrap(chats)
    if resp != None:
        print("\n\n")
        return resp
    else:
        raise HTTPException(
            status_code=400, detail="Not enough members or chats to analyze from 2022!"
        )
