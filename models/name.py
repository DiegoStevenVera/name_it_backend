from pydantic import BaseModel


class Name(BaseModel):
    message: str = 'Default message'
    name: str = 'Default name'