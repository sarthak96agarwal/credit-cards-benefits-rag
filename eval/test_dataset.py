# eval/test_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth evaluation dataset based on 6 credit card benefits PDFs:
#   Amex Gold, Amex Platinum, Amex Delta Gold,
#   Bilt Palladium, Capital One Venture X, United Explorer
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [

    # ── Simple fact retrieval ─────────────────────────────────────────────────

    {
        "question": "What is the maximum auto rental collision damage coverage on the Capital One Venture X?",
        "ground_truth": "The Capital One Venture X Auto Rental Collision Damage Waiver covers up to the actual cash value of rental vehicles with an original MSRP of up to $75,000.",
        "card": "Capital One Venture X",
        "category": "simple_fact",
    },
    {
        "question": "What is the auto rental coverage limit on the United Explorer Card?",
        "ground_truth": "The United Explorer Card Auto Rental Coverage provides reimbursement for damages caused by theft or collision up to $60,000.",
        "card": "United Explorer",
        "category": "simple_fact",
    },
    {
        "question": "What is the maximum cell phone protection per claim on the Bilt Palladium card?",
        "ground_truth": "The Bilt Palladium cell phone protection covers up to $800 per claim, with a $50 deductible, and a maximum of 2 claims per 12-month period.",
        "card": "Bilt Palladium",
        "category": "simple_fact",
    },
    {
        "question": "How much is the annual Uber Cash benefit on the Amex Platinum card?",
        "ground_truth": "The Amex Platinum provides up to $200 per year in Uber Cash, distributed as $15 per month plus a $20 bonus in December.",
        "card": "Amex Platinum",
        "category": "simple_fact",
    },
    {
        "question": "What is the Common Carrier Travel Accident Insurance benefit amount on the United Explorer Card?",
        "ground_truth": "The United Explorer Card provides Common Carrier Travel Accident Insurance up to $500,000 per covered traveler.",
        "card": "United Explorer",
        "category": "simple_fact",
    },

    # ── Exact term matching (tests BM25 / keyword retrieval) ──────────────────

    {
        "question": "Is the auto rental coverage on the United Explorer Card primary or secondary?",
        "ground_truth": "The auto rental coverage on the United Explorer Card is primary coverage.",
        "card": "United Explorer",
        "category": "exact_term",
    },
    {
        "question": "How many Priority Pass lounge guests are included free with the Bilt Palladium card?",
        "ground_truth": "The Bilt Palladium card includes complimentary Priority Pass lounge access for the cardholder and up to 2 guests. Additional guests above 2 are charged $35 per guest per visit.",
        "card": "Bilt Palladium",
        "category": "exact_term",
    },
    {
        "question": "What is the Purchase Protection coverage window on the United Explorer Card?",
        "ground_truth": "The United Explorer Card Purchase Protection covers eligible items within 120 days from the date of purchase, or 90 days for New York residents.",
        "card": "United Explorer",
        "category": "exact_term",
    },

    # ── Benefit details ───────────────────────────────────────────────────────

    {
        "question": "What are the Trip Cancellation and Interruption Insurance limits on the United Explorer Card?",
        "ground_truth": "The United Explorer Card provides Trip Cancellation and Interruption Insurance up to $1,500 per covered traveler and $6,000 per trip for all covered travelers.",
        "card": "United Explorer",
        "category": "benefit_detail",
    },
    {
        "question": "What is the Trip Delay Reimbursement threshold and maximum on the Capital One Venture X?",
        "ground_truth": "The Capital One Venture X Trip Delay Reimbursement applies when the delay exceeds 6 hours or requires an overnight stay. The maximum benefit is $500 per purchased ticket.",
        "card": "Capital One Venture X",
        "category": "benefit_detail",
    },
    {
        "question": "What does the Baggage Delay Insurance cover on the United Explorer Card and what is the maximum benefit?",
        "ground_truth": "The United Explorer Card Baggage Delay Insurance provides $100 per day after the initial 6-hour delay, for each additional 24-hour period, up to a maximum of 3 days. It covers essential personal items like toiletries, a change of clothes, and chargers for electronic devices.",
        "card": "United Explorer",
        "category": "benefit_detail",
    },
    {
        "question": "What is the Lost Luggage Reimbursement maximum on the Capital One Venture X?",
        "ground_truth": "The Capital One Venture X provides Lost Luggage Reimbursement up to $3,000 per covered traveler per trip, with sub-limits of $500 for jewelry and watches, and $500 for cameras and other electronic equipment.",
        "card": "Capital One Venture X",
        "category": "benefit_detail",
    },
    {
        "question": "What dining credits does the Amex Gold card offer?",
        "ground_truth": "The Amex Gold card offers up to $120 per year in dining credits ($10 per month) at participating partners including Grubhub, The Cheesecake Factory, Goldbelly, Wine.com, Five Guys, and participating Shake Shack locations.",
        "card": "Amex Gold",
        "category": "benefit_detail",
    },
    {
        "question": "What is the Airline Fee Credit on the Amex Platinum card?",
        "ground_truth": "The Amex Platinum card offers up to $200 per calendar year in statement credits for incidental airline fees charged by one selected qualifying airline. Enrollment is required and a qualifying airline must be selected.",
        "card": "Amex Platinum",
        "category": "benefit_detail",
    },
    {
        "question": "What are the Trip Cancellation and Interruption limits on the Bilt Palladium card?",
        "ground_truth": "The Bilt Palladium Trip Cancellation and Interruption benefit provides up to $2,000 per covered trip, with a maximum of $5,000 per eligible account per 12-month period.",
        "card": "Bilt Palladium",
        "category": "benefit_detail",
    },
    {
        "question": "What is the Extended Warranty Protection on the United Explorer Card?",
        "ground_truth": "The United Explorer Card Extended Warranty Protection extends the original manufacturer's U.S. repair warranty by one additional year on warranties of 3 years or less. Coverage is limited to $10,000 per item and $50,000 per Account.",
        "card": "United Explorer",
        "category": "benefit_detail",
    },
    {
        "question": "What rewards rate does the Amex Gold card earn at U.S. supermarkets?",
        "ground_truth": "The Amex Gold card earns 4X Membership Rewards points at U.S. supermarkets on up to $25,000 per year in purchases. Superstores, warehouse clubs, and convenience stores are not included.",
        "card": "Amex Gold",
        "category": "benefit_detail",
    },
    {
        "question": "What is the Delta Stays credit on the Delta SkyMiles Gold card?",
        "ground_truth": "The Delta SkyMiles Gold card offers up to $100 per calendar year in statement credits on eligible Delta Stays prepaid hotel or vacation rental bookings on the Delta Stays platform.",
        "card": "Amex Delta Gold",
        "category": "benefit_detail",
    },
    {
        "question": "Does the Delta SkyMiles Gold card offer free checked bags?",
        "ground_truth": "Yes, the Delta SkyMiles Gold card provides the first checked bag free for the Basic Card Member and up to 8 companions on the same reservation on Delta-operated flights.",
        "card": "Amex Delta Gold",
        "category": "benefit_detail",
    },

    # ── Negative questions (answer is no / not available) ─────────────────────

    {
        "question": "Does the Amex Gold card include airport lounge access?",
        "ground_truth": "No, the Amex Gold card does not include airport lounge access.",
        "card": "Amex Gold",
        "category": "negative",
    },
    {
        "question": "Does the United Explorer Card include cell phone protection?",
        "ground_truth": "No, the United Explorer Card does not include cell phone protection.",
        "card": "United Explorer",
        "category": "negative",
    },
    {
        "question": "Does the Delta SkyMiles Gold card include a companion certificate?",
        "ground_truth": "No, the Delta SkyMiles Gold card does not include a companion certificate.",
        "card": "Amex Delta Gold",
        "category": "negative",
    },
    {
        "question": "Does the Capital One Venture X auto rental coverage include motorcycles?",
        "ground_truth": "No, motorcycles are explicitly excluded from the Capital One Venture X Auto Rental Collision Damage Waiver.",
        "card": "Capital One Venture X",
        "category": "negative",
    },
    {
        "question": "Does the Bilt Palladium cell phone protection cover lost phones?",
        "ground_truth": "No, the Bilt Palladium cell phone protection does not cover phones that are lost or mysteriously disappear. It only covers stolen or damaged phones.",
        "card": "Bilt Palladium",
        "category": "negative",
    },

    # ── Cross-card / comparison (hard for naive RAG) ──────────────────────────

    {
        "question": "How does the auto rental coverage limit differ between the Capital One Venture X and the United Explorer Card?",
        "ground_truth": "The Capital One Venture X covers rental vehicles with an MSRP up to $75,000, while the United Explorer Card covers up to $60,000. Both provide primary coverage.",
        "card": "Multiple",
        "category": "cross_card",
    },
    {
        "question": "How does the Trip Delay Reimbursement threshold differ between the Capital One Venture X and the United Explorer Card?",
        "ground_truth": "The Capital One Venture X Trip Delay Reimbursement kicks in after a 6-hour delay, while the United Explorer Card requires a delay of more than 12 hours or an overnight stay. Both offer a maximum benefit of $500.",
        "card": "Multiple",
        "category": "cross_card",
    },
    {
        "question": "Which card has a higher Trip Cancellation limit, the United Explorer Card or the Bilt Palladium?",
        "ground_truth": "The Bilt Palladium has a higher per-trip limit at $2,000 per trip, while the United Explorer Card provides $1,500 per covered traveler. However, the United Explorer allows up to $6,000 per trip for all covered travelers combined.",
        "card": "Multiple",
        "category": "cross_card",
    },
    {
        "question": "Which cards offer Priority Pass lounge access, the Amex Platinum or the Bilt Palladium?",
        "ground_truth": "Both the Amex Platinum and the Bilt Palladium offer Priority Pass lounge access. The Bilt Palladium includes the cardholder and up to 2 guests free. The Amex Platinum includes Priority Pass Select with unlimited access.",
        "card": "Multiple",
        "category": "cross_card",
    },
]
