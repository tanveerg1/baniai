import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import banidb
from app.database import lifespan
from app.nlp import process_punjabi, process_english, detect_intent
from app.recommender import init_recommender, retrain_recommender

app = FastAPI(lifespan=lifespan)
recommender = None

async def cache_shabads(start_id=1, end_id=100):
    shabads_collection = app.mongodb["shabads"]
    for i in range(start_id, end_id + 1):
        try:
            shabad = banidb.shabad(i)
            await shabads_collection.update_one(
                {"shabad_id": shabad["shabadInfo"]["shabadId"]},
                {
                    "$set": {
                        "shabad_id": shabad["shabadInfo"]["shabadId"],
                        "text": " ".join([line["line"]["gurmukhi"]["unicode"] for line in shabad["verses"]]),
                        "translation": " ".join([line["line"]["translation"]["english"]["default"] for line in shabad["verses"]]),
                        "raag": shabad["shabadInfo"]["raag"]["english"],
                        "writer": shabad["shabadInfo"]["writer"]["english"]
                    }
                },
                upsert=True
            )
        except:
            continue


class Query(BaseModel):
    text: str
    language: str  # "pa" or "en"

@app.get("/")
async def root():
    return {"message": "Stock AI API is running!"}

@app.on_event("startup")
async def startup_event():
    global recommender
    recommender = await init_recommender(app.mongodb)
    await cache_shabads(1, 100)

@app.post("/query")
async def process_query(query: Query):
    tokens = process_punjabi(query.text) if query.language == "pa" else process_english(query.text)
    intent = detect_intent(tokens, language=query.language)

    shabads_collection = app.mongodb["shabads"]
    if intent == "search":
        try:
            # Search MongoDB
            results = await shabads_collection.find(
                {"$or": [
                    {"text": {"$regex": "|".join(tokens), "$options": "i"}},
                    {"translation": {"$regex": "|".join(tokens), "$options": "i"}}
                ]}
            ).to_list(length=10)
            if results:
                return results
            # Fallback to banidb
            results = banidb.search({"search_term": query.text})
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif intent == "recommend":
        try:
            last_interaction = await app.mongodb["interactions"].find().sort("timestamp", -1).limit(1).to_list(length=1)
            if not last_interaction:
                return []
            shabad_id = last_interaction[0]["shabad_id"]
            recommendations = recommender.recommend(shabad_id)
            return recommendations
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Could not understand query"}

@app.get("/shabad/{shabad_id}")
async def get_shabad(shabad_id: int):
    shabads_collection = app.mongodb["shabads"]
    shabad = await shabads_collection.find_one({"shabad_id": shabad_id})
    if shabad:
        await app.mongodb["interactions"].insert_one({
            "shabad_id": shabad_id,
            "interaction_type": "view",
            "timestamp": datetime.datetime.utcnow()
        })
        return shabad
    try:
        shabad = banidb.shabad(shabad_id)
        await app.mongodb["interactions"].insert_one({
            "shabad_id": shabad_id,
            "interaction_type": "view",
            "timestamp": datetime.datetime.utcnow()
        })
        return shabad
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/like/{shabad_id}")
async def like_shabad(shabad_id: int):
    await app.mongodb["interactions"].insert_one({
        "shabad_id": shabad_id,
        "interaction_type": "like",
        "timestamp": datetime.datetime.utcnow()
    })
    await retrain_recommender(app.mongodb, recommender)
    return {"message": f"Shabad {shabad_id} liked"}

# if __name__ == '__main__':
#     port = int(os.getenv('PORT', 4000)) 
#     app.run(host='0.0.0.0', port=port, debug=True)