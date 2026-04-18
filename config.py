DIMENSIONS = {
    "model": [
        # Proprietary, large
        "gpt-5.4", "claude-sonnet-4-6",
        # Proprietary, small (within-family size pairs with the large entries above)
        "gpt-5.4-nano", "claude-haiku-4-5",
        # Open weight
        "llama-3.3-70b", "mistral-small",
    ],
    "persona_format": ["bare", "narrative", "third_person", "survey"],
    "question_framing": ["direct", "likert", "forced_choice"],
    "system_prompt": ["neutral", "academic", "survey_admin"],
    "temperature": [0.0, 0.3, 0.7, 1.0],
    "few_shot": [0, 1, 3],
}

# Hierarchical structure for the "deployed system" dimension. Used by
# analysis.hierarchical_system_decomp to break system-level eta-squared
# into provider-level, family-level, and access-type components without
# the open-vs-proprietary confound that the flat decomposition aggregates.
SYSTEM_HIERARCHY = {
    "gpt-5.4":            {"provider": "openai",     "family": "gpt-5.4",   "size": "large", "access": "proprietary"},
    "gpt-5.4-nano":       {"provider": "openai",     "family": "gpt-5.4",   "size": "small", "access": "proprietary"},
    "claude-sonnet-4-6":  {"provider": "anthropic",  "family": "claude-4",  "size": "large", "access": "proprietary"},
    "claude-haiku-4-5":   {"provider": "anthropic",  "family": "claude-4",  "size": "small", "access": "proprietary"},
    "llama-3.3-70b":      {"provider": "meta",       "family": "llama-3.3", "size": "large", "access": "open_weight"},
    "mistral-small":      {"provider": "mistralai",  "family": "mistral-3", "size": "small", "access": "open_weight"},
    # Excluded from primary analyses but kept here for transparency
    "gemini-2.5-flash":   {"provider": "google",     "family": "gemini-2.5","size": "small", "access": "proprietary"},
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
        "scale_max": 5,
        "anes_var": "V241239",
        "expected_gap": "Democrat > Republican",
        "domain": "economic",
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
        "scale_max": 5,
        "anes_var": "V241747",
        "expected_gap": "Democrat > Republican",
        "domain": "cultural",
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
        "scale_max": 5,
        "anes_var": "V242325",
        "expected_gap": "Democrat > Republican",
        "domain": "cultural",
    },
    # === New 7-point ANES policy battery items, added for the v2 multiverse ===
    "defense_spending": {
        "text": "Some people believe that we should spend much less money for defense. Others feel that defense spending should be greatly increased. Where would you place yourself on this scale?",
        "scale": {
            1: "Greatly decrease defense spending",
            2: "Decrease defense spending",
            3: "Decrease defense spending somewhat",
            4: "Keep defense spending about the same",
            5: "Increase defense spending somewhat",
            6: "Increase defense spending",
            7: "Greatly increase defense spending",
        },
        "scale_max": 7,
        "anes_var": "V241242",
        "expected_gap": "Democrat < Republican",  # gap will be negative
        "domain": "economic",
    },
    "healthcare": {
        "text": "There is much concern about the rapid rise in medical and hospital costs. Some people feel there should be a government insurance plan that would cover all medical and hospital expenses for everyone. Others feel that all medical expenses should be paid by individuals through private insurance plans. Where would you place yourself on this scale?",
        "scale": {
            1: "Government insurance plan",
            2: "Mostly government plan",
            3: "Lean toward government plan",
            4: "Mixed plan",
            5: "Lean toward private plan",
            6: "Mostly private plan",
            7: "Private insurance plan",
        },
        "scale_max": 7,
        "anes_var": "V241245",
        "expected_gap": "Democrat < Republican",
        "domain": "economic",
    },
    "abortion": {
        "text": "There has been some discussion about abortion during recent years. Where would you place yourself on this scale?",
        "scale": {
            1: "Abortion should always be permitted without restrictions",
            2: "Abortion should usually be permitted",
            3: "Abortion should be permitted in most cases",
            4: "Abortion should be permitted only in cases such as rape or incest",
            5: "Abortion should rarely be permitted",
            6: "Abortion should usually not be permitted",
            7: "Abortion should never be permitted",
        },
        "scale_max": 7,
        "anes_var": "V241248",
        "expected_gap": "Democrat < Republican",
        "domain": "cultural",
    },
    "guaranteed_jobs": {
        "text": "Some people feel that the government in Washington should see to it that every person has a job and a good standard of living. Others think the government should just let each person get ahead on their own. Where would you place yourself on this scale?",
        "scale": {
            1: "Government should see to jobs and standard of living",
            2: "Mostly government responsibility",
            3: "Lean toward government responsibility",
            4: "Mixed responsibility",
            5: "Lean toward individual responsibility",
            6: "Mostly individual responsibility",
            7: "Government should let each person get ahead on own",
        },
        "scale_max": 7,
        "anes_var": "V241252",
        "expected_gap": "Democrat < Republican",
        "domain": "economic",
    },
    "aid_to_blacks": {
        "text": "Some people feel that the government in Washington should make every effort to improve the social and economic position of Blacks. Others feel that the government should not make any special effort to help Blacks because they should help themselves. Where would you place yourself on this scale?",
        "scale": {
            1: "Government should help Blacks",
            2: "Mostly government should help",
            3: "Lean toward government help",
            4: "Mixed",
            5: "Lean toward Blacks helping themselves",
            6: "Mostly Blacks should help themselves",
            7: "Blacks should help themselves",
        },
        "scale_max": 7,
        "anes_var": "V241255",
        "expected_gap": "Democrat < Republican",
        "domain": "racial",
    },
    "environment_business": {
        "text": "Some people think it is important to protect the environment even if it costs some jobs or otherwise reduces our standard of living. Other people think that protecting the environment is not as important as maintaining jobs and our standard of living. Where would you place yourself on this scale?",
        "scale": {
            1: "Tougher regulations on business needed to protect environment",
            2: "Mostly toughen regulations",
            3: "Lean toward tougher regulations",
            4: "Mixed",
            5: "Lean toward fewer regulations",
            6: "Mostly fewer regulations",
            7: "Regulations to protect environment already too much a burden on business",
        },
        "scale_max": 7,
        "anes_var": "V241258",
        "expected_gap": "Democrat < Republican",
        "domain": "economic",
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
    "defense_spending": [
        {"profile": "A 45-year-old college-educated Democrat from urban California", "response": "3"},
        {"profile": "A 60-year-old Republican from rural Texas with a high school education", "response": "6"},
        {"profile": "A 35-year-old Independent from suburban Ohio with some college", "response": "4"},
    ],
    "healthcare": [
        {"profile": "A 50-year-old college-educated Democrat from New York City", "response": "2"},
        {"profile": "A 55-year-old Republican from rural Alabama with a high school education", "response": "6"},
        {"profile": "A 30-year-old Independent from suburban Colorado with a college degree", "response": "4"},
    ],
    "abortion": [
        {"profile": "A 40-year-old graduate-educated Democrat from urban Massachusetts", "response": "1"},
        {"profile": "A 58-year-old Republican from rural Montana with some college", "response": "6"},
        {"profile": "A 33-year-old Independent from suburban Virginia with a college degree", "response": "3"},
    ],
    "guaranteed_jobs": [
        {"profile": "A 45-year-old college-educated Democrat from urban California", "response": "2"},
        {"profile": "A 60-year-old Republican from rural Texas with a high school education", "response": "6"},
        {"profile": "A 35-year-old Independent from suburban Ohio with some college", "response": "4"},
    ],
    "aid_to_blacks": [
        {"profile": "A 50-year-old college-educated Democrat from New York City", "response": "2"},
        {"profile": "A 55-year-old Republican from rural Alabama with a high school education", "response": "6"},
        {"profile": "A 30-year-old Independent from suburban Colorado with a college degree", "response": "4"},
    ],
    "environment_business": [
        {"profile": "A 40-year-old graduate-educated Democrat from urban Massachusetts", "response": "2"},
        {"profile": "A 58-year-old Republican from rural Montana with some college", "response": "6"},
        {"profile": "A 33-year-old Independent from suburban Virginia with a college degree", "response": "4"},
    ],
}

REPEATS = 5

COST_PER_1M_INPUT = {
    "gpt-5.4-nano": 0.20,
    "gpt-5.4": 2.50,
    "claude-sonnet-4-6": 3.00,
    "claude-haiku-4-5": 1.00,
    "gemini-2.5-flash": 0.30,
    "llama-3.3-70b": 0.39,
    "mistral-small": 0.10,
}

COST_PER_1M_OUTPUT = {
    "gpt-5.4-nano": 0.80,
    "gpt-5.4": 15.00,
    "claude-sonnet-4-6": 15.00,
    "claude-haiku-4-5": 5.00,
    "gemini-2.5-flash": 2.50,
    "llama-3.3-70b": 0.39,
    "mistral-small": 0.40,
}

AVG_INPUT_TOKENS = 400
AVG_OUTPUT_TOKENS = 10
