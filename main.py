# main.py

import logging
from fastapi import FastAPI
from api import product_routes
from fastapi.middleware.cors import CORSMiddleware

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

# Initialize the FastAPI app
app = FastAPI()

origins = [
    "https://azlon.org",
    "https://muchnic.tail9dec88.ts.net",
    #"http://127.0.0.1:8000"
    # Add any other origins if needed
]

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins (you can restrict this to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the product-related routes
app.include_router(product_routes.router)

