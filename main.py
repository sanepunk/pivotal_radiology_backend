from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

from app.core.config import settings
from app.api.routes import auth, patients, analysis
# from app.core.database import connect_to_mongo, close_mongo_connection

# Load environment variables
load_dotenv()

# MongoDB connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MongoDB
    app.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.mongodb = app.mongodb_client[settings.MONGODB_NAME]
    print("Connected to MongoDB")
    yield
    # Close MongoDB connection
    app.mongodb_client.close()
    print("Closed MongoDB connection")

# Create FastAPI app
app = FastAPI(
    title="Pivotal TB Screening API",
    description="Backend API for TB Screening Application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173", "http://localhost:8000", "http://127.0.0.1:8000", "http://127.0.0.1:5173"],  # Frontend URL
    allow_origins="*",  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Add this before your app.mount statements


FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dist1'))
# Mount the uploads directory
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
        <!DOCTYPE html>
    <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/app" />
            <title>Redirecting...</title>
        </head>
        <body>
            <p>Redirecting to application...</p>
            <script>window.location.href = "/app";</script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/patients", tags=["Patients"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])

# Database connection events
# app.add_event_handler("startup", connect_to_mongo)
# app.add_event_handler("shutdown", close_mongo_connection)
# class CORSStaticFiles(StaticFiles):
#     async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
#         if scope["type"] == "http":
#             response = await super().__call__(scope, receive, send)
#             if scope["method"] == "OPTIONS":
#                 response.headers["Access-Control-Allow-Origin"] = "*"
#                 response.headers["Access-Control-Allow-Methods"] = "*"
#                 response.headers["Access-Control-Allow-Headers"] = "*"
#             return response
#         await super().__call__(scope, receive, send)

# # ...existing code...

# # Update your static file mounts to use the new class
# app.mount("/api/v1/files", CORSStaticFiles(directory=UPLOAD_DIR), name="files")
# app.mount("/app", CORSStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

@app.exception_handler(404)
async def spa_fallback(request: Request, exc):
    if request.url.path.startswith("/app"):
        with open(os.path.join(FRONTEND_DIST, "index.html")) as f:
            return HTMLResponse(content=f.read())
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")