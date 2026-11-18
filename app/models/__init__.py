from app.models.shared import AdminActionRun, ModelRun, OddsSnapshot, Prediction
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFightStats,
    UFCFightPrediction, UFCMethodPrediction, UFCFightOdds,
    UFCFightShapValue, UFCFightPreview,
    UFCPredictionMarket, UFCPredictionMarketQuote, UFCPredictionMarketHistory,
)

__all__ = [
    "UFCFighter", "UFCEvent", "UFCFight", "UFCFightStats",
    "UFCFightPrediction", "UFCMethodPrediction", "UFCFightOdds",
    "UFCFightShapValue", "UFCFightPreview",
    "UFCPredictionMarket", "UFCPredictionMarketQuote", "UFCPredictionMarketHistory",
    "Prediction", "ModelRun", "OddsSnapshot", "AdminActionRun",
]

# Feature-track tables (registered for create_all).
from app.models import ufc_fighter_extras, ufc_fightflow, ufc_markets, ufc_officials  # noqa: E402,F401
