"""
Synthetic memory facts used to seed a conversation for manual memory verification.

All facts belong to a single topic — a planned solo trip to Japan — so that
short-term / session memory retrieval can be stress-tested with varied questions
about the same subject (dates, budget, cities, activities, bookings, etc.).

Each fact is tagged with keywords = "synthetic_seed" so the DB can reliably
detect whether a conversation has already been seeded.
"""

SYNTHETIC_SEED_KEYWORD = "synthetic_seed"

SYNTHETIC_FACTS: list[str] = [
    # Trip overview
    "User is planning a solo trip to Japan in October this year.",
    "The trip is 14 days long, starting October 5 and ending October 19.",
    "User is flying from Bangalore (BLR) with a layover in Singapore (SIN).",

    # Cities & itinerary
    "User plans to spend the first 4 days in Tokyo.",
    "After Tokyo, user will travel to Kyoto for 3 days by Shinkansen.",
    "User will then visit Osaka for 2 days before heading to Hiroshima for 1 day.",
    "The final 4 days will be spent back in Tokyo for last-minute shopping and rest.",

    # Accommodation
    "User has already booked a capsule hotel in Akihabara for the Tokyo stays.",
    "In Kyoto, user plans to stay at a traditional ryokan to experience tatami rooms.",
    "Accommodation in Osaka and Hiroshima is not booked yet.",

    # Budget
    "User's total budget for the trip is ₹2,00,000 (approximately ¥300,000).",
    "User has allocated ₹60,000 for flights and ₹80,000 for accommodation.",
    "The remaining ₹60,000 is earmarked for food, transport, and activities.",

    # Transport
    "User will purchase a 14-day JR Pass to cover all Shinkansen travel.",
    "User plans to use the IC card (Suica) for local subway and bus travel.",

    # Activities & interests
    "User wants to visit Tsukiji Outer Market for street food on the first morning.",
    "User has a strong interest in anime and plans to spend a full day in Akihabara.",
    "User wants to hike part of the Fushimi Inari trail at sunrise in Kyoto.",
    "User has booked a seat at a ramen-making class in Tokyo on October 7.",

    # Practical details
    "User is vegetarian and has noted this will be challenging in Japan — has researched shojin ryori (Buddhist veg cuisine) restaurants in Kyoto.",
]
