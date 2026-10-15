#rl_service.py contient la logique d'inférence du modèle.

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import gymnasium as gym

from gymnasium.wrappers import RecordVideo

# Chemin du modèle sauvegardé
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "EAGLE1_TOP_MODEL_PPO_16ENV.zip"
)


class RLService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}"
            )

        # Charger le modèle une seule fois
        self.model = PPO.load(str(MODEL_PATH))

    def predict(self, observation: list[float]) -> list[float]:
        """
        Reçoit les 8 observations de LunarLander
        et retourne les 2 actions continues du modèle PPO.
        """

        obs = np.asarray(observation, dtype=np.float32)

        action, _ = self.model.predict(
            obs,
            deterministic=True
        )

        return action.tolist()

    def play(self, seed: int | None = None) -> dict:
        """
        Exécute un épisode complet de LunarLander.

        Retourne la récompense, la durée et l'historique
        des observations et des actions.
        """

        env = gym.make("LunarLander-v3", continuous=True)

        try:
            obs, _ = env.reset(seed=seed)

            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            history = []

            while not (terminated or truncated):

                action, _ = self.model.predict(
                    obs,
                    deterministic=True
                )

                next_obs, reward, terminated, truncated, _ = env.step(
                    action
                )

                history.append({
                    "step": steps + 1,
                    "observation": obs.tolist(),
                    "action": action.tolist(),
                    "reward": float(reward)
                })

                total_reward += float(reward)
                steps += 1
                obs = next_obs

            return {
                "seed": seed,
                "total_reward": round(total_reward, 2),
                "steps": steps,
                "success": total_reward >= 200,
                "history": history
            }

        finally:
            env.close()

    def generate_video(self, seed: int = 42) -> str:
        """Génère une vidéo MP4 d'un épisode complet."""

        video_dir = BASE_DIR / "outputs" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            render_mode="rgb_array"
        )

        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode: True,
            name_prefix=f"eagle1_seed_{seed}",
            disable_logger=True
        )

        try:
            obs, _ = env.reset(seed=seed)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(
                    obs,
                    deterministic=True
                )

                obs, _, terminated, truncated, _ = env.step(action)

        finally:
            env.close()

        videos = sorted(
            video_dir.glob(f"eagle1_seed_{seed}-episode-*.mp4"),
            key=lambda p: p.stat().st_mtime
        )

        if not videos:
            raise FileNotFoundError("Aucune vidéo générée.")

        return str(videos[-1])

    def evaluate(self, n_episodes: int = 20, start_seed: int = 1000) -> dict:
        """Évalue l'agent sur plusieurs épisodes."""

        results = []

        for i in range(n_episodes):
            seed = start_seed + i

            episode = self.play(seed=seed)

            # Conserver uniquement les métriques utiles
            results.append({
                "episode": i + 1,
                "seed": seed,
                "total_reward": episode["total_reward"],
                "steps": episode["steps"],
                "success": episode["success"],
                "history": episode["history"]
            })

        rewards = np.array(
            [episode["total_reward"] for episode in results]
        )

        return {
            "n_episodes": n_episodes,
            "mean_reward": round(float(rewards.mean()), 2),
            "std_reward": round(float(rewards.std()), 2),
            "success_rate": round(
                float(np.mean(rewards >= 200) * 100), 2
            ),
            "episodes": results
        }