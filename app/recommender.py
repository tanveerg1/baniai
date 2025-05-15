from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import banidb
import pandas as pd
import datetime

async def load_shabads(mongodb):
    shabads = await mongodb["shabads"].find().to_list(length=1000)
    return pd.DataFrame(shabads)

def preprocess_features(df):
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    text_features = tfidf.fit_transform(df["translation"])
    enc = OneHotEncoder()
    categorical_features = enc.fit_transform(df[["raag", "writer"]]).toarray()
    return np.hstack([text_features.toarray(), categorical_features])

class ShabadRecommender:
    def __init__(self, features, shabads_df):
        self.features = features
        self.shabads_df = shabads_df

    def recommend(self, shabad_id, top_n=3):
        idx = self.shabads_df.index[self.shabads_df["shabad_id"] == shabad_id].tolist()
        if not idx:
            return []
        idx = idx[0]
        sim_scores = cosine_similarity([self.features[idx]], self.features)[0]
        sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
        return self.shabads_df.iloc[sim_indices][["shabad_id", "text", "translation", "raag", "writer"]].to_dict("records")

async def init_recommender(mongodb):
    df = await load_shabads(mongodb)
    features = preprocess_features(df)
    return ShabadRecommender(features, df)

async def retrain_recommender(mongodb, recommender):
    interactions = await mongodb["interactions"].find().to_list(length=1000)
    liked_shabads = pd.Series([i["shabad_id"] for i in interactions if i["interaction_type"] == "like"]).value_counts()
    weights = np.ones(len(recommender.shabads_df))
    for shabad_id, count in liked_shabads.items():
        idx = recommender.shabads_df.index[recommender.shabads_df["shabad_id"] == shabad_id].tolist()
        if idx:
            weights[idx[0]] += count * 0.2
    weighted_features = recommender.features * weights[:, None]
    recommender.features = weighted_features

async def cache_shabad(shabad_id, mongodb):
    shabad = banidb.shabad(shabad_id)
    await mongodb["shabads"].update_one(
        {"shabad_id": shabad["shabadInfo"]["shabadId"]},
        {"$set": {
            "shabad_id": shabad["shabadInfo"]["shabadId"],
            "text": " ".join([line["line"]["gurmukhi"]["unicode"] for line in shabad["verses"]]),
            "translation": " ".join([line["line"]["translation"]["english"]["default"] for line in shabad["verses"]]),
            "raag": shabad["shabadInfo"]["raag"]["english"],
            "writer": shabad["shabadInfo"]["writer"]["english"]
        }},
        upsert=True
    )

async def get_random_shabad(mongodb):
    shabad = banidb.random()
    await cache_shabad(shabad["shabadInfo"]["shabadId"], mongodb)
    return shabad

async def search_shabads(query, language, mongodb):
    if language == "pa":
        results = banidb.search({"search_term": query, "search_type": 3})
    else:
        results = banidb.search({"search_term": query, "search_type": 3})
    # Cache results in MongoDB
    for result in results["results"]:
        await cache_shabad(result["shabadInfo"]["shabadId"], mongodb)
    return results["results"]

async def get_filters():
    return {
        "raags": banidb.raags(),
        "writers": banidb.writers(),
        "sources": banidb.sources()
    }

async def log_interaction(shabad_id, interaction_type, mongodb):
    await mongodb["interactions"].insert_one({
        "shabad_id": shabad_id,
        "interaction_type": interaction_type,
        "timestamp": datetime.datetime.utcnow()
    })