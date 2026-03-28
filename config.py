DIMENSIONS = {
    "model": ["gpt-5.4-nano", "gpt-5.4", "claude-sonnet-4-6", "gemini-2.5-flash", "llama-3.3-70b", "mistral-small"],
    "persona_format": ["bare", "narrative", "third_person", "survey"],
    "question_framing": ["direct", "likert", "forced_choice"],
    "system_prompt": ["neutral", "academic", "survey_admin"],
    "temperature": [0.0, 0.3, 0.7, 1.0],
    "few_shot": [0, 1, 3],
}

DIMENSION_ORDER = [
    "model",
    "persona_format",
    "question_framing",
    "system_prompt",
    "temperature",
    "few_shot",
]

PROFILES = [
    {"id": "PA_D1", "state": "PA", "party": "Democrat", "age": 32, "education": "college", "race": "White", "gender": "Female", "area": "suburban Philadelphia"},
    {"id": "PA_R1", "state": "PA", "party": "Republican", "age": 55, "education": "high school", "race": "White", "gender": "Male", "area": "rural Westmoreland County"},
    {"id": "PA_D2", "state": "PA", "party": "Democrat", "age": 45, "education": "graduate", "race": "Black", "gender": "Male", "area": "Pittsburgh"},
    {"id": "PA_R2", "state": "PA", "party": "Republican", "age": 38, "education": "some college", "race": "White", "gender": "Female", "area": "Lehigh Valley"},
    {"id": "MI_D1", "state": "MI", "party": "Democrat", "age": 28, "education": "college", "race": "Black", "gender": "Female", "area": "Detroit"},
    {"id": "MI_R1", "state": "MI", "party": "Republican", "age": 62, "education": "high school", "race": "White", "gender": "Male", "area": "rural Upper Peninsula"},
    {"id": "MI_D2", "state": "MI", "party": "Democrat", "age": 41, "education": "graduate", "race": "White", "gender": "Male", "area": "Ann Arbor"},
    {"id": "MI_R2", "state": "MI", "party": "Republican", "age": 50, "education": "college", "race": "White", "gender": "Female", "area": "Grand Rapids suburbs"},
    {"id": "WI_D1", "state": "WI", "party": "Democrat", "age": 35, "education": "college", "race": "Hispanic", "gender": "Male", "area": "Milwaukee"},
    {"id": "WI_R1", "state": "WI", "party": "Republican", "age": 58, "education": "some college", "race": "White", "gender": "Male", "area": "rural Waukesha County"},
    {"id": "WI_D2", "state": "WI", "party": "Democrat", "age": 24, "education": "college", "race": "White", "gender": "Female", "area": "Madison"},
    {"id": "WI_R2", "state": "WI", "party": "Republican", "age": 47, "education": "high school", "race": "White", "gender": "Female", "area": "Green Bay"},
    {"id": "AZ_D1", "state": "AZ", "party": "Democrat", "age": 39, "education": "college", "race": "Hispanic", "gender": "Female", "area": "Phoenix"},
    {"id": "AZ_R1", "state": "AZ", "party": "Republican", "age": 65, "education": "college", "race": "White", "gender": "Male", "area": "Scottsdale"},
    {"id": "AZ_D2", "state": "AZ", "party": "Democrat", "age": 30, "education": "some college", "race": "Native American", "gender": "Male", "area": "Tucson"},
    {"id": "AZ_R2", "state": "AZ", "party": "Republican", "age": 44, "education": "graduate", "race": "White", "gender": "Female", "area": "Mesa"},
    {"id": "GA_D1", "state": "GA", "party": "Democrat", "age": 33, "education": "college", "race": "Black", "gender": "Female", "area": "Atlanta"},
    {"id": "GA_R1", "state": "GA", "party": "Republican", "age": 57, "education": "high school", "race": "White", "gender": "Male", "area": "rural South Georgia"},
    {"id": "GA_D2", "state": "GA", "party": "Democrat", "age": 26, "education": "graduate", "race": "Black", "gender": "Male", "area": "Savannah"},
    {"id": "GA_R2", "state": "GA", "party": "Republican", "age": 42, "education": "college", "race": "White", "gender": "Female", "area": "Cobb County suburbs"},
]

ANES_ITEMS = {
    "gov_spending": {
        "text": "Some people think the government should provide fewer services, even in areas such as health and education, in order to reduce spending. Other people feel that it is important for the government to provide many more services even if it means an increase in spending. Where would you place yourself on this scale?",
        "scale": {
            1: "Government should provide many fewer services; reduce spending a lot",
            2: "Reduce services and spending somewhat",
            3: "Keep services and spending about the same",
            4: "Increase services and spending somewhat",
            5: "Government should provide many more services; increase spending a lot",
        },
        "expected_gap": "Democrat > Republican",
    },
    "immigration": {
        "text": "Do you think the number of immigrants from foreign countries who are permitted to come to the United States to live should be increased a lot, increased a little, left the same as it is now, decreased a little, or decreased a lot?",
        "scale": {
            1: "Decreased a lot",
            2: "Decreased a little",
            3: "Left the same",
            4: "Increased a little",
            5: "Increased a lot",
        },
        "expected_gap": "Democrat > Republican",
    },
    "gun_control": {
        "text": "Do you think the federal government should make it more difficult for people to buy a gun than it is now, make it easier for people to buy a gun, or keep these rules about the same as they are now?",
        "scale": {
            1: "Make it much easier",
            2: "Make it somewhat easier",
            3: "Keep the same",
            4: "Make it somewhat more difficult",
            5: "Make it much more difficult",
        },
        "expected_gap": "Democrat > Republican",
    },
}

SYSTEM_PROMPTS = {
    "neutral": "You are participating in a research study. Please respond honestly based on the perspective described.",
    "academic": "You are assisting academic researchers studying American public opinion. Please provide responses that accurately reflect how the described individual would likely respond to a political survey. Respond with only the number corresponding to your answer.",
    "survey_admin": "You are a survey respondent completing a nationally representative political attitudes survey. Answer as the person described. Give only your answer number, no explanation.",
}

FEW_SHOT_EXAMPLES = {
    "gov_spending": [
        {"profile": "A 45-year-old college-educated Democrat from urban California", "response": "4"},
        {"profile": "A 60-year-old Republican from rural Texas with a high school education", "response": "2"},
        {"profile": "A 35-year-old Independent from suburban Ohio with some college", "response": "3"},
    ],
    "immigration": [
        {"profile": "A 50-year-old college-educated Democrat from New York City", "response": "4"},
        {"profile": "A 55-year-old Republican from rural Alabama with a high school education", "response": "1"},
        {"profile": "A 30-year-old Independent from suburban Colorado with a college degree", "response": "3"},
    ],
    "gun_control": [
        {"profile": "A 40-year-old graduate-educated Democrat from urban Massachusetts", "response": "5"},
        {"profile": "A 58-year-old Republican from rural Montana with some college", "response": "2"},
        {"profile": "A 33-year-old Independent from suburban Virginia with a college degree", "response": "3"},
    ],
}

REPEATS = 5

COST_PER_1M_INPUT = {
    "gpt-5.4-nano": 0.20,
    "gpt-5.4": 2.50,
    "claude-sonnet-4-6": 3.00,
    "gemini-2.5-flash": 0.30,
    "llama-3.3-70b": 0.39,
    "mistral-small": 0.10,
}

COST_PER_1M_OUTPUT = {
    "gpt-5.4-nano": 0.80,
    "gpt-5.4": 15.00,
    "claude-sonnet-4-6": 15.00,
    "gemini-2.5-flash": 2.50,
    "llama-3.3-70b": 0.39,
    "mistral-small": 0.40,
}

AVG_INPUT_TOKENS = 400
AVG_OUTPUT_TOKENS = 10
