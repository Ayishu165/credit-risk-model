from pydantic import BaseModel

class PredictionInput(BaseModel):
    Amount: float
    Value: float
    CountryCode: int
    PricingStrategy: int
    CurrencyCode_UGX: int
    ProductCategory_airtime: int
    ProductCategory_financial_services: int
    ProductCategory_utility_bill: int
    ChannelId_ChannelId_2: int
    ChannelId_ChannelId_3: int

class PredictionOutput(BaseModel):
    risk_score: float
