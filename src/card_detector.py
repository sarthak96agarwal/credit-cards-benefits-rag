"""
LLM-based card name detection.

Given a user question, identifies which credit card(s) the question is about
by calling gpt-4o-mini with structured JSON output.
"""

import json
import openai
from dotenv import load_dotenv

load_dotenv()

# Known card names — must match what index.py stores in metadata
CARD_NAMES = [
    "Amex Gold",
    "Amex Platinum",
    "Amex Delta Gold",
    "Bilt Palladium",
    "Capital One Venture X",
    "United Explorer",
]

_oai_client = openai.OpenAI()

CARD_DETECT_PROMPT = f"""You are a card name detector. Given a user question about credit card benefits, identify which card(s) the question is about.

Available cards: {json.dumps(CARD_NAMES)}

Rules:
- Return ONLY card names from the list above.
- If the question mentions multiple cards, return all of them.
- If the question is general and not about any specific card, return an empty list.
- Match common aliases: "Venture X" → "Capital One Venture X", "Bilt" → "Bilt Palladium", "Delta Gold" / "SkyMiles Gold" → "Amex Delta Gold", "Platinum" → "Amex Platinum", "Gold card" → "Amex Gold", etc.

Return a JSON array of card names. Examples:
- "What lounge access does the Venture X have?" → ["Capital One Venture X"]
- "Compare trip delay between Bilt and United Explorer" → ["Bilt Palladium", "United Explorer"]
- "Which card has the best dining rewards?" → []"""


def detect_cards(question: str) -> list[str]:
    """Detect which card(s) a question is about using an LLM."""
    response = _oai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CARD_DETECT_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    result = json.loads(response.choices[0].message.content)
    # Handle both {"cards": [...]} and [...] formats
    cards = result if isinstance(result, list) else result.get("cards", [])
    # Validate against known card names
    return [c for c in cards if c in CARD_NAMES]
