from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# Initializing FastAPI server
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading up the Trained Model
model = pickle.load(open("./diseases_trained_model.sav", "rb"))

# Defining the Model Input Types
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

# Setting up the home route
@app.get("/")
def read_root():
    return {"Data": "Welcome to Rabbit Diseases Prediction Model"}

# Setting up the Prediction Route
@app.post("/prediction")
async def get_predict(data: Disease):
    sample = [[
        data.appetite,
        data.digestion,
        data.defecation,
        data.posture,
        data.lethargy,
        data.scratching,
        data.hd_shaking,
        data.crust_area,
        data.drooling,
        data.swollen_area,
        data.red_ears,
        data.balance,
        data.hd_lifting,
        data.sneezing,
        data.watery_eyes,
        data.runny_nose,
        data.breathing,
        data.hd_titl,
        data.peeing_frq,
        data.peeing_sludge,
        data.peeing_bloody,
        data.weght_loss,
        data.discharged_eyes,
        data.overgwn_teeth,
        data.lump,
        data.fur_lost
    ]]
    result = model.predict(sample).tolist()[0]
    return {
        "data": {
            "result": result,
            "interpretation": "GI Stasis" if result == 1 else "Not GI Stasis"
        }
    }