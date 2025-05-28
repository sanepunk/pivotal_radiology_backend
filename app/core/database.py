from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "pivotal")

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect_db(cls):
        if cls.client is None:
            cls.client = AsyncIOMotorClient(MONGODB_URL)
            cls.db = cls.client[DATABASE_NAME]

    @classmethod
    async def close_db(cls):
        if cls.client is not None:
            cls.client.close()
            cls.client = None
            cls.db = None

    @classmethod
    def get_db(cls):
        return cls.db

def get_database():
    if Database.db is None:
        Database.client = AsyncIOMotorClient(MONGODB_URL)
        Database.db = Database.client[DATABASE_NAME]
    return Database.db 