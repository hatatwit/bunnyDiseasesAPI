from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn

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
abscesses_model = pickle.load(open("./abscesses_model.sav", "rb"))
bladder_sludge_model = pickle.load(open("./bladder_sludge_model.sav", "rb"))
ear_mites_model = pickle.load(open("./ear_mites_model.sav", "rb"))
fur_mites_model = pickle.load(open("./fur_mites_model.sav", "rb"))
gi_stasis_model = pickle.load(open("./gi_stasis_model.sav", "rb"))
heatstroke_model = pickle.load(open("./heatstroke_model.sav", "rb"))
overgrown_teeth_model = pickle.load(open("./overgrown_teeth_model.sav", "rb"))
snuffles_model = pickle.load(open("./snuffles_model.sav", "rb"))

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

    result = {
        "Abscesses": abscesses_model.predict(sample).tolist()[0],
        "Bladder sludge": bladder_sludge_model.predict(sample).tolist()[0],
        "Ear mites": ear_mites_model.predict(sample).tolist()[0],
        "Fur mites": fur_mites_model.predict(sample).tolist()[0],
        "GI stasis": gi_stasis_model.predict(sample).tolist()[0],
        "Heatstroke": heatstroke_model.predict(sample).tolist()[0],
        "Overgrown teeth": overgrown_teeth_model.predict(sample).tolist()[0],
        "Snuffles": snuffles_model.predict(sample).tolist()[0]
    }
    return {
        "data": {
            "result": result,
            "interpretation": [key for key, value in result.items() if value == 1]
        }
    }

if __name__ == "__main__":

    uvicorn.run(app, port=8000, host="localhost")