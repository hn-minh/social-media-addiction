from typing import Literal
from pydantic import BaseModel, Field

CountryType = Literal[
    'Bangladesh', 'India', 'USA', 'UK', 'Canada', 'Australia', 'Germany', 'Brazil',
    'Japan', 'South Korea', 'France', 'Spain', 'Italy', 'Mexico', 'Russia', 'China',
    'Sweden', 'Norway', 'Denmark', 'Netherlands', 'Belgium', 'Switzerland',
    'Austria', 'Portugal', 'Greece', 'Ireland', 'New Zealand', 'Singapore',
    'Malaysia', 'Thailand', 'Vietnam', 'Philippines', 'Indonesia', 'Taiwan',
    'Hong Kong', 'Turkey', 'Israel', 'UAE', 'Egypt', 'Morocco', 'South Africa',
    'Nigeria', 'Kenya', 'Ghana', 'Argentina', 'Chile', 'Colombia', 'Peru',
    'Venezuela', 'Ecuador', 'Uruguay', 'Paraguay', 'Bolivia', 'Costa Rica',
    'Panama', 'Jamaica', 'Trinidad', 'Bahamas', 'Iceland', 'Finland', 'Poland',
    'Romania', 'Hungary', 'Czech Republic', 'Slovakia', 'Croatia', 'Serbia',
    'Slovenia', 'Bulgaria', 'Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Moldova',
    'Belarus', 'Kazakhstan', 'Uzbekistan', 'Kyrgyzstan', 'Tajikistan', 'Armenia',
    'Georgia', 'Azerbaijan', 'Cyprus', 'Malta', 'Luxembourg', 'Monaco', 'Andorra',
    'San Marino', 'Vatican City', 'Liechtenstein', 'Montenegro', 'Albania',
    'North Macedonia', 'Kosovo', 'Bosnia', 'Qatar', 'Kuwait', 'Bahrain', 'Oman',
    'Jordan', 'Lebanon', 'Iraq', 'Yemen', 'Syria', 'Afghanistan', 'Pakistan',
    'Nepal', 'Bhutan', 'Sri Lanka', 'Maldives'
]

PlatformType = Literal[
    'Instagram', 'Twitter', 'TikTok', 'YouTube', 'Facebook', 'LinkedIn', 'Snapchat',
    'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
]

class UserInput(BaseModel):
    Age: int = Field(gt=0, le=120, description="Tuổi người dùng")
    Gender: Literal["Male", "Female"]
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Country: CountryType
    Avg_Daily_Usage_Hours: float = Field(ge=0.0, le=24.0)
    Most_Used_Platform: PlatformType
    Affects_Academic_Performance: bool
    Sleep_Hours_Per_Night: float = Field(ge=0.0, le=12.0)
    Mental_Health_Score: int = Field(ge=1, le=10)
    Relationship_Status: Literal["Single", "In Relationship", "Complicated"]
    Conflicts_Over_Social_Media: int = Field(ge=1, le=10)