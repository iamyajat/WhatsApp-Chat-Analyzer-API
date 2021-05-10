from fastapi import FastAPI

app = FastAPI(
    title="WhatsApp Analyzer",
    version="1.0",
    description="Get beautiful insights about your chats!",
)

@app.get("/")
async def root():
    return {"message": "Service is up!"}