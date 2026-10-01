from pymongo import MongoClient
from config import MONGODB_URL

client = MongoClient(MONGODB_URL)

try:
    db = client["store_logs"]
    collection = db["db_collection"]
except:
    print("mongo-db is not working")

