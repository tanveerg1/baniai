from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

# Get MongoDB credentials from environment variables
db_username = os.getenv("MONGO_DB_USERNAME")
db_password = os.getenv("MONGO_DB_PASSWORD")
cluster_name = os.getenv("MONGO_DB_CLUSTER_NAME")
app_name = os.getenv("MONGO_DB_APP_NAME")   

MONGODB_URI = f"mongodb+srv://{db_username}:{db_password}@baniaidb.{cluster_name}.mongodb.net/"  # Replaced by env var on Render

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    app.mongodb = app.mongodb_client["baniaidb"]
    print("MongoDB connected")
    yield
    app.mongodb_client.close()
    print("MongoDB disconnected")  