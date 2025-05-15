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

# Cache shabads
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

# Cache angs
async def cache_ang(ang, source="G"):
    angs_collection = app.mongodb["angs"]
    try:
        ang_data = banidb.ang(ang, source)
        await angs_collection.update_one(
            {"ang": ang, "source": source},
            {
                "$set": {
                    "ang": ang_data["pageNo"],
                    "source": ang_data["source"]["id"],
                    "verses": [
                        {
                            "line_id": verse["line"]["lineId"],
                            "gurmukhi": verse["line"]["gurmukhi"]["unicode"],
                            "translation": verse["line"]["translation"]["english"]["default"],
                            "page_no": verse["line"]["pageNo"]
                        } for verse in ang_data["verses"]
                    ]
                }
            },
            upsert=True
        )
        return ang_data
    except:
        return None

# Cache metadata
async def cache_metadata():
    metadata_collection = app.mongodb["metadata"]
    try:
        raags = banidb.raags()
        writers = banidb.writers()
        sources = banidb.sources()
        await metadata_collection.update_one(
            {"type": "raags"},
            {"$set": {"type": "raags", "data": raags}},
            upsert=True
        )
        await metadata_collection.update_one(
            {"type": "writers"},
            {"$set": {"type": "writers", "data": writers}},
            upsert=True
        )
        await metadata_collection.update_one(
            {"type": "sources"},
            {"$set": {"type": "sources", "data": sources}},
            upsert=True
        )
    except:
        pass

class Query(BaseModel):
    text: str
    language: str  # "pa" or "en"

@app.get("/")
async def root():
    return {"message": "Bani AI API is running!"}

@app.on_event("startup")
async def startup_event():
    global recommender
    recommender = await init_recommender(app.mongodb)
    await cache_shabads(1, 100)
    await cache_metadata()
    # Create indexes for performance
    await app.mongodb["shabads"].create_index("shabad_id", unique=True)
    await app.mongodb["angs"].create_index([("ang", 1), ("source", 1)], unique=True)
    await app.mongodb["interactions"].create_index("shabad_id")
    await app.mongodb["metadata"].create_index("type", unique=True)

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
            for result in results.get("results", []):
                await cache_shabads(result["shabadInfo"]["shabadId"], result["shabadInfo"]["shabadId"])
            return results.get("results", [])
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
        await cache_shabads(shabad_id, shabad_id)
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

@app.get("/ang/{ang}")
async def get_ang(ang: int, source: str = "G"):
    angs_collection = app.mongodb["angs"]
    ang_data = await angs_collection.find_one({"ang": ang, "source": source})
    if ang_data:
        await app.mongodb["interactions"].insert_one({
            "ang": ang,
            "source": source,
            "interaction_type": "view_ang",
            "timestamp": datetime.datetime.utcnow()
        })
        return ang_data
    try:
        ang_data = await cache_ang(ang, source)
        if not ang_data:
            raise Exception("Failed to fetch ang")
        await app.mongodb["interactions"].insert_one({
            "ang": ang,
            "source": source,
            "interaction_type": "view_ang",
            "timestamp": datetime.datetime.utcnow()
        })
        return ang_data
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/random")
async def get_random_shabad(source: str = "G"):
    try:
        shabad = banidb.random(source)
        shabad_id = shabad["shabadInfo"]["shabadId"]
        await cache_shabads(shabad_id, shabad_id)
        await app.mongodb["interactions"].insert_one({
            "shabad_id": shabad_id,
            "interaction_type": "view",
            "timestamp": datetime.datetime.utcnow()
        })
        return shabad
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
async def get_metadata():
    metadata_collection = app.mongodb["metadata"]
    try:
        raags = await metadata_collection.find_one({"type": "raags"})
        writers = await metadata_collection.find_one({"type": "writers"})
        sources = await metadata_collection.find_one({"type": "sources"})
        return {
            "raags": raags["data"] if raags else banidb.raags(),
            "writers": writers["data"] if writers else banidb.writers(),
            "sources": sources["data"] if sources else banidb.sources()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == '__main__':
#     port = int(os.getenv('PORT', 4000)) 
#     app.run(host='0.0.0.0', port=port, debug=True)