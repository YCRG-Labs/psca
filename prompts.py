from config import ANES_ITEMS, SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES


def build_persona(profile, persona_format):
    if persona_format == "bare":
        return (
            f"You are a {profile['age']}-year-old {profile['race']} "
            f"{profile['gender']} {profile['party']} from "
            f"{profile['area']}, {profile['state']} with a "
            f"{profile['education']} education."
        )

    if persona_format == "narrative":
        pronoun = "She" if profile["gender"] == "Female" else "He"
        poss = "her" if profile["gender"] == "Female" else "his"
        return (
            f"Imagine you are a {profile['age']}-year-old "
            f"{profile['race'].lower()} {profile['gender'].lower()} "
            f"living in {profile['area']}, {profile['state']}. "
            f"{pronoun} completed {poss} education at the "
            f"{profile['education']} level and has consistently "
            f"identified with the {profile['party']} party. "
            f"{pronoun} follows politics regularly and votes in "
            f"most elections."
        )

    if persona_format == "third_person":
        pronoun = "She" if profile["gender"] == "Female" else "He"
        poss = "Her" if profile["gender"] == "Female" else "His"
        return (
            f"The respondent is a {profile['age']}-year-old "
            f"{profile['race'].lower()} {profile['gender'].lower()} "
            f"who lives in {profile['area']}, {profile['state']}. "
            f"{poss} highest level of education is {profile['education']}. "
            f"{pronoun} identifies as a {profile['party']} and has voted "
            f"consistently with the party in recent elections. "
            f"Answer the following question as this person would."
        )

    return (
        f"Respondent demographics:\n"
        f"  Age: {profile['age']}\n"
        f"  Gender: {profile['gender']}\n"
        f"  Race/Ethnicity: {profile['race']}\n"
        f"  Party Identification: {profile['party']}\n"
        f"  Education: {profile['education']}\n"
        f"  Location: {profile['area']}, {profile['state']}"
    )


def build_question(item_key, question_framing):
    item = ANES_ITEMS[item_key]
    scale_max = item.get("scale_max", max(item["scale"].keys()))

    if question_framing == "direct":
        scale_text = "\n".join(
            f"  {k}. {v}" for k, v in item["scale"].items()
        )
        return (
            f"{item['text']}\n\n"
            f"Choose a number from 1-{scale_max}:\n{scale_text}\n\n"
            f"Respond with only the number."
        )

    if question_framing == "likert":
        return (
            f"On a scale of 1 to {scale_max}:\n"
            f"  1 = {item['scale'][1]}\n"
            f"  {scale_max} = {item['scale'][scale_max]}\n\n"
            f"{item['text']}\n\n"
            f"Respond with only a number from 1 to {scale_max}."
        )

    choices = "\n".join(
        f"  ({chr(96 + k)}) {v}" for k, v in item["scale"].items()
    )
    return (
        f"{item['text']}\n\n"
        f"Which comes closest to your view?\n{choices}\n\n"
        f"Respond with only the letter."
    )


def build_few_shot(item_key, n_shots, question_framing):
    if n_shots == 0:
        return ""

    examples = FEW_SHOT_EXAMPLES[item_key][:n_shots]
    lines = ["Here are some example responses from other respondents:\n"]
    for ex in examples:
        if question_framing == "forced_choice":
            response = chr(96 + int(ex["response"]))
        else:
            response = ex["response"]
        lines.append(f"  {ex['profile']} → {response}")
    lines.append("\nNow provide your response.\n")
    return "\n".join(lines)


def build_prompt(spec, profile, item_key):
    persona = build_persona(profile, spec["persona_format"])
    question = build_question(item_key, spec["question_framing"])
    few_shot = build_few_shot(
        item_key, spec["few_shot"], spec["question_framing"]
    )
    system = SYSTEM_PROMPTS[spec["system_prompt"]]

    return {
        "system": system,
        "user": f"{persona}\n\n{few_shot}{question}",
        "model": spec["model"],
        "temperature": spec["temperature"],
    }
