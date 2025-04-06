# main.py

import logging
import os
from fastapi import FastAPI, Depends
from api import product_routes
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up global logging configuration
# This will log to both the console and a file
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO or DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named 'app.log'
        logging.StreamHandler()  # Also log to console
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Check for required environment variables
if not os.environ.get('JINA_API_KEY'):
    logger.warning("JINA_API_KEY not found in environment variables. API calls will fail.")

# Check for MongoDB connection string
if not os.environ.get('MONGODB_URI'):
    logger.warning("MONGODB_URI not found in environment variables. Using default local connection.")
    # For local development - you should set the real connection string in .env
    os.environ['MONGODB_URI'] = "mongodb://mongo:27017/facecrime"

# Initialize the FastAPI app
app = FastAPI(
    title="FaceCrime Backend API",
    description="API for detecting similar faces using Jina embeddings and MongoDB",
    version="0.1.0"
)

origins = [
    "https://facecrime.io",
    "https://muchnic.tail9dec88.ts.net",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    # Add any other origins if needed
]

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the product-related routes
app.include_router(product_routes.router)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "FaceCrime API is running"}

