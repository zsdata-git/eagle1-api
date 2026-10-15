from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rl_service import RLService

from fastapi.responses import FileResponse


# Initialisation de l'API
app = FastAPI(
    title="Eagle-1 RL API",
    description="API de prédiction du pilote automatique LunarLander",
    version="1.0.0"
)

# Chargement du modèle PPO
rl_service = RLService()


# Schéma des données entrantes
class ObservationInput(BaseModel):
    observation: list[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="État LunarLander composé de 8 valeurs"
    )


# Vérification du fonctionnement de l'API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "EAGLE1_TOP_MODEL_PPO_16ENV"
    }


# Prédiction d'une action
@app.post("/predict")
def predict(data: ObservationInput):
    action = rl_service.predict(data.observation)

    return {
        "action": action
    }

# Simulation d'une partie complète
class PlayInput(BaseModel):
    seed: int | None = Field(
        default=None,
        description="Graine aléatoire pour reproduire une partie"
    )


@app.post("/play")
def play(data: PlayInput):
    return rl_service.play(seed=data.seed)

@app.get("/video")
def generate_video(seed: int = 42):
    video_path = rl_service.generate_video(seed=seed)

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"eagle1_seed_{seed}.mp4"
    )

class EvaluationInput(BaseModel):
    n_episodes: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Nombre d'épisodes à évaluer (1 à 100)"
    )

    start_seed: int = Field(
        default=1000,
        ge=0,
        description="Première seed utilisée pour l'évaluation"
    )


@app.post("/evaluate")
def evaluate(data: EvaluationInput):
    return rl_service.evaluate(
        n_episodes=data.n_episodes,
        start_seed=data.start_seed
    )