from pathlib import Path
from dotenv import load_dotenv
import os
import random


class APPENV():
    def __init__(self):
        current = Path.cwd()
        while current != current.parent:
            env_path = current / "config" / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded: {env_path}")
                break
            current = current.parent
        else:
            raise FileNotFoundError("Could not find .env file in any parent directory.")

        load_dotenv(env_path)
        self.host = os.getenv("HOST")
        self.port = int(os.getenv("PORT"))
        self.client_id = random.randint(1, 100)