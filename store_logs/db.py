from pymongo import MongoClient
import rag.config as config

client = MongoClient(config.MONGODB_URL)

try:
    db = client["store_logs"]
    collection = db["db_collection"]
except:
    print("mongo-db is not working")

