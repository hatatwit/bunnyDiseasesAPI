from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

models = {}
diseases = [
    "Abscesses", "Bladder sludge", "Ear mites", "Fur mites",
    "GI stasis", "Heatstroke", "Overgrown teeth", "Snuffles"
]

for disease in diseases:
    models[disease] = pickle.load(open(f"./{disease.lower().replace(' ', '_')}_model.sav", "rb"))

class Disease(BaseModel):
    appetite: int
    digestion: int
    defecation: int
    posture: int
    lethargy: int
    scratching: int
    hd_shaking: int
    crust_area: int
    drooling: int
    swollen_area: int
    red_ears: int
    balance: int
    hd_lifting: int
    sneezing: int
    watery_eyes: int
    runny_nose: int
    breathing: int
    hd_titl: int
    peeing_frq: int
    peeing_sludge: int
    peeing_bloody: int
    weght_loss: int
    discharged_eyes: int
    overgwn_teeth: int
    lump: int
    fur_lost: int

@app.get("/")
def read_root():
    return {"Data": "Welcome to Rabbit Diseases Prediction Model"}

@app.post("/prediction")
async def get_predict(data: Disease):
    sample = [[getattr(data, attribute) for attribute in dir(data) if not attribute.startswith('_')]]

    result = {disease: models[disease].predict(sample).tolist()[0] for disease in diseases}

    return {
        "data": {
            "result": result,
            "interpretation": [key for key, value in result.items() if value == 1]
        }
    }
