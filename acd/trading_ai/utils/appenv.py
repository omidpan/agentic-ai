from pathlib import Path
from dotenv import load_dotenv
import os
import random
import logging
logger = logging.getLogger(__name__)
###### This is relative path and need to be handled correctly when running the script from different locations.
# The following line changes the current working directory to the script's directory.
os.chdir('<from / to learningibt directory>')
class APPENV():
    '''
    Add os.chdir(<from / to learningibt>) to the top of your script to ensure
    that the current working directory is set correctly. This is important for relative paths to work as expected.
    '''
    
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
        self.client_id = random.randint(1, 10000)