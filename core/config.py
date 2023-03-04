from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/API/V1"
    PROJECT_NAME: str = "NAME IT"
    FILES_PATH = 'files'
    TOKENS: object = None
    PET_NAME_MODEL: object = None
    RESNET_MODEL: object = None

    class Config:
        env_file = ".env"

settings = Settings()