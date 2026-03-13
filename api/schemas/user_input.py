from pydantic import BaseModel

class UserInput(BaseModel):
    Age: int
    Gender: str
    Academic_Level: str
    Country: str
    Avg_Daily_Usage_Hours: float
    Most_Used_Platform: str
    Affects_Academic_Performance: bool
    Sleep_Hours_Per_Night: float
    Mental_Health_Score: int
    Relationship_Status: str
    Conflicts_Over_Social_Media: int