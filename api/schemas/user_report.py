from pydantic import BaseModel
from api.schemas.user_input import UserInput

class UserReport(UserInput):
    Addicted_Score: int