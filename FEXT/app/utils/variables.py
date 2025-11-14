from __future__ import annotations

import os

from dotenv import load_dotenv

from FEXT.app.utils.constants import PROJECT_DIR
from FEXT.app.utils.logger import logger
from FEXT.app.utils.singleton import singleton


# [LOAD ENVIRONMENT VARIABLES]
###############################################################################
@singleton
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = os.path.join(PROJECT_DIR, "setup", ".env")
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path, override=True)
        else:
            logger.error(f".env file not found at: {self.env_path}")

    # -------------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)
    
env_variables = EnvironmentVariables()