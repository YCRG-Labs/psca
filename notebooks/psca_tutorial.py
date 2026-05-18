# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=5.5",
#     "marimo",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///
import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="P-SCA Tutorial")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompt Specification Curve Analysis

    *A reproducible tour of how arbitrary prompt choices drive LLM-simulated public opinion.*

    When a researcher asks a language model to simulate a survey respondent, the resulting "public opinion" depends not only on the population that has been described but on a dozen small decisions about how to ask. This notebook is an applied tour of P-SCA, a framework for treating those decisions as the object of study rather than as nuisance parameters. By the time you reach the end you will have constructed prompts interactively, sampled a multiverse of plausible designs, watched a specification curve come together in real time, and seen the single configuration in which a 70-billion-parameter model reports that Republicans want stricter gun laws than Democrats.

    Everything that follows runs on a corpus of 160,035 valid model responses collected in May 2026, shipped alongside this notebook, so no API keys are required.

    > Companion paper: Crainic, J., Yee, B., and Koh, P. (2026), *Prompt Specification Curve Analysis: How Arbitrary Choices Drive LLM-Simulated Opinion*. Toolkit at [`YCRG-Labs/psca`](https://github.com/YCRG-Labs/psca).
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt

    PALETTE = {
        "blue":      "#0F62FE",
        "red":       "#DA1E28",
        "ink":       "#161616",
        "graydark":  "#6F6F6F",
        "graymute":  "#A8A8A8",
        "graylight": "#F4F4F4",
    }
    SERIF = "Charter, 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif"

    def _ycrg_theme():
        return {
            "config": {
                "background": "white",
                "view": {"stroke": None, "continuousWidth": 360, "continuousHeight": 240},
                "axis": {
                    "labelFont": SERIF, "titleFont": SERIF,
                    "labelFontSize": 11, "titleFontSize": 12,
                    "labelColor": PALETTE["ink"], "titleColor": PALETTE["ink"],
                    "domainColor": PALETTE["graydark"], "domainWidth": 0.6,
                    "tickColor": PALETTE["graydark"], "tickWidth": 0.5, "tickSize": 4,
                    "gridColor": PALETTE["graylight"], "gridWidth": 0.5,
                    "labelPadding": 4, "titlePadding": 8,
                },
                "header": {
                    "labelFont": SERIF, "titleFont": SERIF,
                    "labelFontSize": 11, "titleFontSize": 12,
                    "labelColor": PALETTE["ink"], "titleColor": PALETTE["ink"],
                },
                "legend": {
                    "labelFont": SERIF, "titleFont": SERIF,
                    "labelFontSize": 11, "titleFontSize": 12,
                    "labelColor": PALETTE["ink"], "titleColor": PALETTE["ink"],
                    "symbolStrokeWidth": 0, "padding": 6,
                },
                "title": {
                    "font": SERIF, "fontSize": 14, "fontWeight": "normal",
                    "color": PALETTE["ink"], "anchor": "start", "offset": 8,
                },
                "range": {
                    "category": [
                        PALETTE["blue"], PALETTE["red"], PALETTE["graydark"],
                        PALETTE["graymute"], "#393939", "#525252",
                    ],
                },
                "bar": {"binSpacing": 0, "discreteBandSize": {"band": 0.85}},
            }
        }

    if hasattr(alt, "theme") and "register" in dir(alt.theme):
        alt.theme.register("ycrg", enable=True)(_ycrg_theme)
    else:
        alt.themes.register("ycrg", _ycrg_theme)
        alt.themes.enable("ycrg")

    def _load_text(filename):
        try:
            import urllib.request
            url = str(mo.notebook_location() / "data" / filename)
            if url.startswith("http://") or url.startswith("https://"):
                with urllib.request.urlopen(url) as r:
                    return r.read().decode("utf-8")
            from pathlib import Path
            return Path(url).read_text()
        except Exception:
            from pathlib import Path
            local = Path(__file__).parent / "data" / filename
            return local.read_text() if local.exists() else ""

    PIPELINE_SVG = _load_text("pipeline.svg")
    DATA_URL = str(mo.notebook_location() / "data" / "multiverse_results.csv.gz")

    return DATA_URL, PALETTE, PIPELINE_SVG, alt, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The 30-second finding
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before any methodology, here is the headline that the rest of the notebook unpacks. Holding the model, the item, and the population fixed, simply swapping the question framing from a direct multiple-choice presentation to a Likert anchor turns Llama 3.3 70B's gun-control gap from a healthy positive number into a negative one. The same model, in the same multiverse, with one knob moved, publishes a different finding.
    """)
    return


@app.cell
def _(PALETTE, mo, pd):
    _cold = pd.DataFrame([
        {"framing": "direct",         "Democrat mean": 4.62, "Republican mean": 2.34, "gap (D − R)": "+2.28"},
        {"framing": "forced-choice",  "Democrat mean": 4.51, "Republican mean": 2.50, "gap (D − R)": "+2.01"},
        {"framing": "likert",         "Democrat mean": 1.99, "Republican mean": 3.91, "gap (D − R)": "−1.92"},
    ])
    _stripe = (
        f'<div style="border-left:3px solid {PALETTE["red"]}; '
        f'background:{PALETTE["graylight"]}; padding:10px 14px; '
        f'font-family:{"Charter, Iowan Old Style, Palatino, Georgia, serif"}; '
        f'color:{PALETTE["ink"]}; margin-bottom:10px">'
        f'<b style="color:{PALETTE["red"]}">A single prompt knob flips the partisan sign.</b> '
        f'Llama 3.3 70B, gun control, 20 ANES profiles, 5 repeats per cell.'
        f'</div>'
    )
    mo.vstack([
        mo.Html(_stripe),
        mo.ui.table(_cold, selection=None, page_size=5),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The fragility problem
    """)
    return


@app.cell(hide_code=True)
def _(PIPELINE_SVG, mo):
    if PIPELINE_SVG:
        mo.Html(
            f'<div style="margin:6px 0 18px 0; overflow-x:auto; '
            f'text-align:center; display:flex; justify-content:center">'
            f'<div style="max-width:100%">{PIPELINE_SVG}</div></div>'
        )
    else:
        mo.md("*Pipeline schematic unavailable; rebuild with `pdflatex figures/pipeline_schematic.tex && pdftocairo -svg figures/pipeline_schematic.pdf notebooks/data/pipeline.svg`.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Silicon sampling, the practice of using a language model to stand in for a survey respondent, has become an attractive shortcut across political science, marketing, and human-computer interaction. The appeal is straightforward: a few thousand carefully written prompts can, in principle, do the work of a hundred-thousand-dollar survey panel. What the literature has been slower to confront is that the prompt itself is underspecified. Whether the persona is rendered as a bare demographic vignette or as a short narrative biography, whether the question is asked in its original ANES wording or compressed into a Likert anchor, whether the system message tells the model it is helping with academic research or filling out a survey, whether the temperature is set close to zero or closer to one, whether examples accompany the question or not, every one of these choices is plausibly defensible, and together they describe a multiverse of equally reasonable designs that no single paper has the page count to fully report.

    P-SCA borrows specification curve analysis from econometrics, where Simonsohn, Simmons, and Nelson (2020) developed it to expose the sensitivity of empirical estimates to analytical choices, and rebuilds it for the prompt-design problem. The empirical question becomes whether the partisan signal that motivates a silicon-sampling result survives the rest of the multiverse, or whether the headline depends on a small region of design space that the authors happened to occupy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Six prompt dimensions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every specification in the P-SCA multiverse is a point in a six-dimensional grid whose axes correspond to the choices most often left implicit in published work. The widget below lets you assemble any point in that grid and see exactly what the API would receive, which makes the abstraction concrete in a way that prose cannot. Change a single dropdown and watch the prompt rewrite itself; the goal is to develop the intuition that "the same question" is, in operational terms, a family of different questions.
    """)
    return


@app.cell(hide_code=True)
def _():
    DIMENSIONS = {
        "model": [
            "gpt-5.4", "claude-sonnet-4-6",
            "gpt-5.4-nano", "claude-haiku-4-5",
            "llama-3.3-70b", "mistral-small",
        ],
        "persona_format":   ["bare", "narrative", "third_person", "survey"],
        "question_framing": ["direct", "likert", "forced_choice"],
        "system_prompt":    ["neutral", "academic", "survey_admin"],
        "temperature":      [0.0, 0.3, 0.7, 1.0],
        "few_shot":         [0, 1, 3],
    }

    PROFILES = [
        {"id": "PA_D1", "state": "PA", "party": "Democrat",   "age": 32, "education": "college",      "race": "White",           "gender": "Female", "area": "suburban Philadelphia"},
        {"id": "PA_R1", "state": "PA", "party": "Republican", "age": 55, "education": "high school",  "race": "White",           "gender": "Male",   "area": "rural Westmoreland County"},
        {"id": "PA_D2", "state": "PA", "party": "Democrat",   "age": 45, "education": "graduate",     "race": "Black",           "gender": "Male",   "area": "Pittsburgh"},
        {"id": "PA_R2", "state": "PA", "party": "Republican", "age": 38, "education": "some college", "race": "White",           "gender": "Female", "area": "Lehigh Valley"},
        {"id": "MI_D1", "state": "MI", "party": "Democrat",   "age": 28, "education": "college",      "race": "Black",           "gender": "Female", "area": "Detroit"},
        {"id": "MI_R1", "state": "MI", "party": "Republican", "age": 62, "education": "high school",  "race": "White",           "gender": "Male",   "area": "rural Upper Peninsula"},
        {"id": "MI_D2", "state": "MI", "party": "Democrat",   "age": 41, "education": "graduate",     "race": "White",           "gender": "Male",   "area": "Ann Arbor"},
        {"id": "MI_R2", "state": "MI", "party": "Republican", "age": 50, "education": "college",      "race": "White",           "gender": "Female", "area": "Grand Rapids suburbs"},
        {"id": "WI_D1", "state": "WI", "party": "Democrat",   "age": 35, "education": "college",      "race": "Hispanic",        "gender": "Male",   "area": "Milwaukee"},
        {"id": "WI_R1", "state": "WI", "party": "Republican", "age": 58, "education": "some college", "race": "White",           "gender": "Male",   "area": "rural Waukesha County"},
        {"id": "WI_D2", "state": "WI", "party": "Democrat",   "age": 24, "education": "college",      "race": "White",           "gender": "Female", "area": "Madison"},
        {"id": "WI_R2", "state": "WI", "party": "Republican", "age": 47, "education": "high school",  "race": "White",           "gender": "Female", "area": "Green Bay"},
        {"id": "AZ_D1", "state": "AZ", "party": "Democrat",   "age": 39, "education": "college",      "race": "Hispanic",        "gender": "Female", "area": "Phoenix"},
        {"id": "AZ_R1", "state": "AZ", "party": "Republican", "age": 65, "education": "college",      "race": "White",           "gender": "Male",   "area": "Scottsdale"},
        {"id": "AZ_D2", "state": "AZ", "party": "Democrat",   "age": 30, "education": "some college", "race": "Native American", "gender": "Male",   "area": "Tucson"},
        {"id": "AZ_R2", "state": "AZ", "party": "Republican", "age": 44, "education": "graduate",     "race": "White",           "gender": "Female", "area": "Mesa"},
        {"id": "GA_D1", "state": "GA", "party": "Democrat",   "age": 33, "education": "college",      "race": "Black",           "gender": "Female", "area": "Atlanta"},
        {"id": "GA_R1", "state": "GA", "party": "Republican", "age": 57, "education": "high school",  "race": "White",           "gender": "Male",   "area": "rural South Georgia"},
        {"id": "GA_D2", "state": "GA", "party": "Democrat",   "age": 26, "education": "graduate",     "race": "Black",           "gender": "Male",   "area": "Savannah"},
        {"id": "GA_R2", "state": "GA", "party": "Republican", "age": 42, "education": "college",      "race": "White",           "gender": "Female", "area": "Cobb County suburbs"},
    ]

    ANES_ITEMS = {
        "gov_spending": {
            "text": "Some people think the government should provide fewer services, even in areas such as health and education, in order to reduce spending. Other people feel that it is important for the government to provide many more services even if it means an increase in spending. Where would you place yourself on this scale?",
            "scale": {1: "Government should provide many fewer services; reduce spending a lot", 2: "Reduce services and spending somewhat", 3: "Keep services and spending about the same", 4: "Increase services and spending somewhat", 5: "Government should provide many more services; increase spending a lot"},
            "scale_max": 5, "expected_gap": "Democrat > Republican",
        },
        "immigration": {
            "text": "Do you think the number of immigrants from foreign countries who are permitted to come to the United States to live should be increased a lot, increased a little, left the same as it is now, decreased a little, or decreased a lot?",
            "scale": {1: "Decreased a lot", 2: "Decreased a little", 3: "Left the same", 4: "Increased a little", 5: "Increased a lot"},
            "scale_max": 5, "expected_gap": "Democrat > Republican",
        },
        "gun_control": {
            "text": "Do you think the federal government should make it more difficult for people to buy a gun than it is now, make it easier for people to buy a gun, or keep these rules about the same as they are now?",
            "scale": {1: "Make it much easier", 2: "Make it somewhat easier", 3: "Keep the same", 4: "Make it somewhat more difficult", 5: "Make it much more difficult"},
            "scale_max": 5, "expected_gap": "Democrat > Republican",
        },
    }

    SYSTEM_PROMPTS = {
        "neutral":      "You are participating in a research study. Please respond honestly based on the perspective described.",
        "academic":     "You are assisting academic researchers studying American public opinion. Please provide responses that accurately reflect how the described individual would likely respond to a political survey. Respond with only the number corresponding to your answer.",
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

    def _persona(profile, fmt):
        if fmt == "bare":
            return (f"You are a {profile['age']}-year-old {profile['race']} "
                    f"{profile['gender']} {profile['party']} from "
                    f"{profile['area']}, {profile['state']} with a "
                    f"{profile['education']} education.")
        if fmt == "narrative":
            pronoun = "She" if profile["gender"] == "Female" else "He"
            poss    = "her" if profile["gender"] == "Female" else "his"
            return (f"Imagine you are a {profile['age']}-year-old "
                    f"{profile['race'].lower()} {profile['gender'].lower()} "
                    f"living in {profile['area']}, {profile['state']}. "
                    f"{pronoun} completed {poss} education at the "
                    f"{profile['education']} level and has consistently "
                    f"identified with the {profile['party']} party. "
                    f"{pronoun} follows politics regularly and votes in most elections.")
        if fmt == "third_person":
            pronoun = "She" if profile["gender"] == "Female" else "He"
            poss    = "Her" if profile["gender"] == "Female" else "His"
            return (f"The respondent is a {profile['age']}-year-old "
                    f"{profile['race'].lower()} {profile['gender'].lower()} "
                    f"who lives in {profile['area']}, {profile['state']}. "
                    f"{poss} highest level of education is {profile['education']}. "
                    f"{pronoun} identifies as a {profile['party']} and has voted "
                    f"consistently with the party in recent elections. "
                    f"Answer the following question as this person would.")
        return (f"Respondent demographics:\n"
                f"  Age: {profile['age']}\n"
                f"  Gender: {profile['gender']}\n"
                f"  Race/Ethnicity: {profile['race']}\n"
                f"  Party Identification: {profile['party']}\n"
                f"  Education: {profile['education']}\n"
                f"  Location: {profile['area']}, {profile['state']}")

    def _question(item_key, framing):
        item = ANES_ITEMS[item_key]
        scale_max = item["scale_max"]
        if framing == "direct":
            scale_text = "\n".join(f"  {k}. {v}" for k, v in item["scale"].items())
            return (f"{item['text']}\n\nChoose a number from 1-{scale_max}:\n{scale_text}\n\nRespond with only the number.")
        if framing == "likert":
            return (f"On a scale of 1 to {scale_max}:\n"
                    f"  1 = {item['scale'][1]}\n"
                    f"  {scale_max} = {item['scale'][scale_max]}\n\n"
                    f"{item['text']}\n\nRespond with only a number from 1 to {scale_max}.")
        choices = "\n".join(f"  ({chr(96 + k)}) {v}" for k, v in item["scale"].items())
        return (f"{item['text']}\n\nWhich comes closest to your view?\n{choices}\n\nRespond with only the letter.")

    def _fewshot(item_key, n_shots, framing):
        if n_shots == 0:
            return ""
        examples = FEW_SHOT_EXAMPLES[item_key][:n_shots]
        lines = ["Here are some example responses from other respondents:\n"]
        for ex in examples:
            if framing == "forced_choice":
                response = chr(96 + int(ex["response"]))
            else:
                response = ex["response"]
            lines.append(f"  {ex['profile']} → {response}")
        lines.append("\nNow provide your response.\n")
        return "\n".join(lines)

    def build_prompt(spec, profile, item_key):
        return {
            "system": SYSTEM_PROMPTS[spec["system_prompt"]],
            "user":   f"{_persona(profile, spec['persona_format'])}\n\n"
                      f"{_fewshot(item_key, spec['few_shot'], spec['question_framing'])}"
                      f"{_question(item_key, spec['question_framing'])}",
            "model":       spec["model"],
            "temperature": spec["temperature"],
        }

    return ANES_ITEMS, DIMENSIONS, PROFILES, build_prompt


@app.cell
def _(ANES_ITEMS, DIMENSIONS, PROFILES, mo):
    model_dd      = mo.ui.dropdown(DIMENSIONS["model"], value="claude-sonnet-4-6", label="model")
    persona_dd    = mo.ui.dropdown(DIMENSIONS["persona_format"], value="narrative", label="persona format")
    framing_dd    = mo.ui.dropdown(DIMENSIONS["question_framing"], value="direct", label="question framing")
    sysprompt_dd  = mo.ui.dropdown(DIMENSIONS["system_prompt"], value="neutral", label="system prompt")
    temp_dd       = mo.ui.dropdown([str(t) for t in DIMENSIONS["temperature"]], value="0.7", label="temperature")
    fewshot_dd    = mo.ui.dropdown([str(n) for n in DIMENSIONS["few_shot"]], value="0", label="few-shot")
    profile_dd    = mo.ui.dropdown([p["id"] for p in PROFILES], value="PA_R1", label="profile")
    item_dd       = mo.ui.dropdown(list(ANES_ITEMS.keys()), value="gun_control", label="ANES item")
    mo.vstack([
        mo.hstack([model_dd, persona_dd, framing_dd, sysprompt_dd], justify="center"),
        mo.hstack([temp_dd, fewshot_dd, profile_dd, item_dd], justify="center"),
    ], align="center")
    return (
        fewshot_dd,
        framing_dd,
        item_dd,
        model_dd,
        persona_dd,
        profile_dd,
        sysprompt_dd,
        temp_dd,
    )


@app.cell
def _(
    PALETTE,
    PROFILES,
    build_prompt,
    fewshot_dd,
    framing_dd,
    item_dd,
    mo,
    model_dd,
    persona_dd,
    profile_dd,
    sysprompt_dd,
    temp_dd,
):
    _spec = {
        "model":            model_dd.value,
        "persona_format":   persona_dd.value,
        "question_framing": framing_dd.value,
        "system_prompt":    sysprompt_dd.value,
        "temperature":      float(temp_dd.value),
        "few_shot":         int(fewshot_dd.value),
    }
    _profile = next(p for p in PROFILES if p["id"] == profile_dd.value)
    _prompt = build_prompt(_spec, _profile, item_dd.value)
    _block = (
        f'<div style="max-width:760px; margin:8px auto; '
        f'font-family:Charter, \'Iowan Old Style\', Palatino, Georgia, serif; '
        f'color:{PALETTE["ink"]}">'
        f'<div style="font-size:12px; letter-spacing:0.05em; text-transform:uppercase; '
        f'color:{PALETTE["graydark"]}; margin-bottom:4px">System prompt</div>'
        f'<pre style="background:{PALETTE["graylight"]}; padding:10px 12px; '
        f'border-radius:3px; white-space:pre-wrap; font-size:12.5px; margin:0 0 12px 0">'
        f'{_prompt["system"]}</pre>'
        f'<div style="font-size:12px; letter-spacing:0.05em; text-transform:uppercase; '
        f'color:{PALETTE["graydark"]}; margin-bottom:4px">User prompt</div>'
        f'<pre style="background:{PALETTE["graylight"]}; padding:10px 12px; '
        f'border-radius:3px; white-space:pre-wrap; font-size:12.5px; margin:0 0 10px 0">'
        f'{_prompt["user"]}</pre>'
        f'<div style="font-size:12.5px; color:{PALETTE["graydark"]}">'
        f'API call: <code>{_prompt["model"]}</code> at temperature '
        f'<code>{_prompt["temperature"]}</code></div>'
        f'</div>'
    )
    mo.Html(_block)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Stacking the cardinalities of those six dimensions gives 2,592 unique prompt configurations before any consideration of which voter is being simulated or which ANES item is being asked. Folding in twenty battleground-state profiles, three items, and five repeats per cell pushes the full factorial north of three quarters of a million API calls, which is wasteful even at nano-tier prices. The next section describes the design that makes a meaningful audit affordable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Latin Hypercube sampling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rather than enumerate the factorial, P-SCA draws from it using Latin Hypercube sampling, a quasi-random scheme that guarantees roughly balanced coverage along every marginal dimension while randomizing the joint structure. The practical effect is that six hundred LHS draws cover the design space more uniformly than six hundred independent random draws would, especially along the lower-cardinality axes. The slider lets you watch coverage materialize as the sample size grows, with the bars indicating how often each level of each dimension is hit.
    """)
    return


@app.cell(hide_code=True)
def _(DIMENSIONS):
    from scipy.stats.qmc import LatinHypercube

    _DIM_ORDER = ["model", "persona_format", "question_framing",
                  "system_prompt", "temperature", "few_shot"]

    def generate_specifications(n_samples, seed=42):
        sampler = LatinHypercube(d=len(_DIM_ORDER), seed=seed)
        draws = sampler.random(n=n_samples)
        specs = []
        for i, row in enumerate(draws):
            spec = {"spec_id": i}
            for j, dim in enumerate(_DIM_ORDER):
                levels = DIMENSIONS[dim]
                idx = min(int(row[j] * len(levels)), len(levels) - 1)
                spec[dim] = levels[idx]
            specs.append(spec)
        return specs

    return (generate_specifications,)


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(50, 600, value=300, step=50, label="n specifications")
    mo.hstack([n_samples_slider], justify="center")
    return (n_samples_slider,)


@app.cell
def _(PALETTE, alt, generate_specifications, mo, n_samples_slider, pd):
    _specs = generate_specifications(n_samples=n_samples_slider.value, seed=42)
    _specs_df = pd.DataFrame(_specs)
    _counts = (
        _specs_df.melt(id_vars="spec_id", var_name="dimension", value_name="level")
        .groupby(["dimension", "level"], observed=True).size()
        .reset_index(name="count")
    )
    _lhs_chart = (
        alt.Chart(_counts)
        .mark_bar(color=PALETTE["blue"], opacity=0.85)
        .encode(
            x=alt.X("level:N", title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=9)),
            y=alt.Y("count:Q", title="# specs"),
        )
        .properties(width=105, height=120)
        .facet(column=alt.Column("dimension:N", title=None, header=alt.Header(labelFontSize=11), spacing=8))
        .resolve_scale(x="independent")
        .configure_facet(spacing=8)
    )
    mo.vstack([
        mo.md(f"**{len(_specs):,} specifications drawn from LHS.** Marginal coverage by dimension:"),
        _lhs_chart,
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each sampled specification is then evaluated on twenty profiles, three items, and five repeats, which yields three hundred API calls per spec. The published P-SCA run uses six hundred LHS specs spread across six model systems, producing one hundred and eighty thousand calls of which 160,035 returned parseable responses. That entire dataset is what the rest of this notebook draws on, and you can run every analysis below without ever touching an API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The multiverse results
    """)
    return


@app.cell
def _(DATA_URL, mo, pd):
    try:
        df = pd.read_csv(DATA_URL, compression="gzip")
        for _c in ["party", "item", "model", "persona_format",
                   "question_framing", "system_prompt", "profile_id"]:
            df[_c] = df[_c].astype("category")
    except Exception:
        df = pd.DataFrame()
    if len(df) == 0:
        _summary = mo.md("**Data file not found.** Expected `notebooks/data/multiverse_results.csv.gz` or `multiverse_results.parquet`. Rebuild from `results/full_lhs.json` if you have it locally.")
    else:
        _summary = mo.md(
            f"""
    The shipped artifact contains **{len(df):,} valid responses** spread across **{df['spec_id'].nunique()} specifications**, **{df['model'].nunique()} model systems**, **{df['item'].nunique()} ANES items**, and **{df['profile_id'].nunique()} simulated profiles**.

    For the headline numbers reported in the paper we drop Gemini 2.5 Flash, whose parse-failure rate on direct and Likert framings ran between ten and fourteen percent and would have contaminated downstream variance estimates (see paper §4.1 for the diagnostic). The next cell applies that exclusion so that the analyses that follow line up with the published figures.
    """
        )
    _summary
    return (df,)


@app.cell
def _(df, mo):
    df_main = df[df["model"] != "gemini-2.5-flash"].copy()
    mo.md(f"After excluding Gemini: **{len(df_main):,} responses** across **{df_main['spec_id'].nunique()} specifications**.")
    return (df_main,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Pick your own headline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Suppose you were writing a single-shot silicon-sampling paper. You would settle on a model, a persona format, a question framing, a system prompt, a temperature, and a number of few-shot examples, and you would report whatever partisan gap fell out of that configuration. The widget below lets you play out that thought experiment against real data: choose any combination of the six dimensions, and the notebook will go looking for matching rows in the published P-SCA run and tell you what gap a paper built on that exact prompt would have reported.
    """)
    return


@app.cell
def _(DIMENSIONS, mo):
    pick_model    = mo.ui.dropdown(DIMENSIONS["model"],            value="llama-3.3-70b",     label="model")
    pick_persona  = mo.ui.dropdown(DIMENSIONS["persona_format"],   value="narrative",         label="persona format")
    pick_framing  = mo.ui.dropdown(DIMENSIONS["question_framing"], value="likert",            label="question framing")
    pick_sysp     = mo.ui.dropdown(DIMENSIONS["system_prompt"],    value="neutral",           label="system prompt")
    pick_temp     = mo.ui.dropdown([str(t) for t in DIMENSIONS["temperature"]], value="0.7", label="temperature")
    pick_fewshot  = mo.ui.dropdown([str(n) for n in DIMENSIONS["few_shot"]],    value="0",   label="few-shot")
    pick_item     = mo.ui.dropdown(["gov_spending", "immigration", "gun_control"], value="gun_control", label="item")
    mo.vstack([
        mo.hstack([pick_model, pick_persona, pick_framing, pick_sysp], justify="center"),
        mo.hstack([pick_temp, pick_fewshot, pick_item], justify="center"),
    ], align="center")
    return (
        pick_fewshot,
        pick_framing,
        pick_item,
        pick_model,
        pick_persona,
        pick_sysp,
        pick_temp,
    )


@app.cell
def _(
    PALETTE,
    df,
    mo,
    pick_fewshot,
    pick_framing,
    pick_item,
    pick_model,
    pick_persona,
    pick_sysp,
    pick_temp,
):
    _mask = (
        (df["model"]            == pick_model.value)    &
        (df["persona_format"]   == pick_persona.value)  &
        (df["question_framing"] == pick_framing.value)  &
        (df["system_prompt"]    == pick_sysp.value)     &
        (df["temperature"]      == float(pick_temp.value)) &
        (df["few_shot"]         == int(pick_fewshot.value)) &
        (df["item"]             == pick_item.value)
    )
    _matched = df[_mask]
    if len(_matched) == 0:
        _verdict = mo.md(
            "**No specification in the six-hundred-cell LHS run lands on this exact combination.** "
            "That absence is itself informative, because even six hundred LHS draws "
            "cover only a fraction of the 2,592 possible cells; nudging any one "
            "dimension will usually land on a sampled neighbor."
        )
    else:
        _per_party = _matched.groupby("party", observed=True)["score"].mean()
        _d = _per_party.get("Democrat", float("nan"))
        _r = _per_party.get("Republican", float("nan"))
        _gap = _d - _r
        if _gap > 0.3:
            _word, _color = "consistent with the expected partisan direction", PALETTE["blue"]
        elif _gap < -0.3:
            _word, _color = "an inverted result, where the partisan sign flips", PALETTE["red"]
        else:
            _word, _color = "essentially flat, with no detectable partisan signal", PALETTE["graydark"]
        _verdict = mo.md(
            f"""
    **The chosen specification matches {_matched['spec_id'].nunique()} cell(s) and {len(_matched)} responses.**

    | | mean response |
    |---|---|
    | Democrat | {_d:.2f} |
    | Republican | {_r:.2f} |
    | **Gap (D minus R)** | <span style="color:{_color}; font-weight:600">{_gap:+.2f}, {_word}</span> |

    Had you settled on exactly this prompt design and reported it as the headline of your paper, the number you would have written is the one above.
    """
        )
    mo.hstack([_verdict], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Some combinations worth trying. Selecting Llama 3.3 70B with the Likert framing on gun control reproduces the inversion that the cold open opened with. Switching the same item to Claude Sonnet 4.6 with the direct framing recovers a calm gap of roughly two scale points in the expected direction. Choosing GPT-5.4-nano with forced-choice framing lands somewhere in between. The data has not changed, the question has not changed, and yet the six knobs alone span the distance between a sober finding and an embarrassing one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Live specification curve
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The specification curve generalizes the previous demo from a single chosen cell to every cell at once. For each spec we compute the difference between mean Democratic and mean Republican responses, sign-adjusted so that movement in the politically expected direction is positive, and then sort the specs from most negative to most positive. Bars colored blue lie on the expected side of zero; bars colored red represent specifications that would have reported the wrong-sign partisan gap. The filters below let you re-shape the multiverse on the fly. Dropping a model, restricting to a single framing, or narrowing the temperature window causes the curve, the percentage of specifications on the expected side, and the displayed range to re-render together, which makes the dependence of the headline on each choice immediately visible.
    """)
    return


@app.cell
def _(DIMENSIONS, df_main, mo):
    curve_item_dd = mo.ui.dropdown(
        sorted(df_main["item"].unique()), value="gun_control", label="ANES item"
    )
    model_filter = mo.ui.multiselect(
        options=sorted(df_main["model"].unique()),
        value=sorted(df_main["model"].unique()),
        label="include models",
    )
    framing_filter = mo.ui.multiselect(
        options=DIMENSIONS["question_framing"],
        value=DIMENSIONS["question_framing"],
        label="include framings",
    )
    persona_filter = mo.ui.multiselect(
        options=DIMENSIONS["persona_format"],
        value=DIMENSIONS["persona_format"],
        label="include persona formats",
    )
    temp_range = mo.ui.range_slider(
        start=0.0, stop=1.0, step=0.1, value=(0.0, 1.0), label="temperature range",
    )
    mo.vstack([
        mo.hstack([curve_item_dd, temp_range], justify="center"),
        mo.hstack([model_filter], justify="center"),
        mo.hstack([framing_filter], justify="center"),
        mo.hstack([persona_filter], justify="center"),
    ], align="center")
    return (
        curve_item_dd,
        framing_filter,
        model_filter,
        persona_filter,
        temp_range,
    )


@app.cell
def _(
    PALETTE,
    alt,
    curve_item_dd,
    df_main,
    framing_filter,
    mo,
    model_filter,
    pd,
    persona_filter,
    temp_range,
):
    _filtered = df_main[
        (df_main["item"] == curve_item_dd.value) &
        (df_main["model"].isin(model_filter.value)) &
        (df_main["question_framing"].isin(framing_filter.value)) &
        (df_main["persona_format"].isin(persona_filter.value)) &
        (df_main["temperature"].between(temp_range.value[0], temp_range.value[1]))
    ]
    if len(_filtered) == 0:
        _curve_out = mo.md("**The current filter selects no rows.** Re-enable a model, framing, or persona format to populate the curve.")
    else:
        _means = (
            _filtered.groupby(["spec_id", "party"], observed=True)["score"]
            .mean().reset_index()
        )
        _wide = _means.pivot_table(index="spec_id", columns="party", values="score").reset_index()
        _wide["gap"] = _wide["Democrat"] - _wide["Republican"]
        _wide = _wide.dropna(subset=["gap"]).sort_values("gap").reset_index(drop=True)
        _wide["rank"] = range(len(_wide))
        _wide["sign"] = (_wide["gap"] > 0).map({True: "expected", False: "inverted"})
        _pct_expected = 100 * (_wide["gap"] > 0).mean()

        _bars = (
            alt.Chart(_wide)
            .mark_bar(size=2)
            .encode(
                x=alt.X("rank:Q", title="Specification (sorted by gap)"),
                y=alt.Y("gap:Q", title="Partisan gap (D minus R)"),
                color=alt.Color(
                    "sign:N",
                    scale=alt.Scale(domain=["expected", "inverted"],
                                    range=[PALETTE["blue"], PALETTE["red"]]),
                    legend=alt.Legend(title=None, orient="top-right"),
                ),
                tooltip=["spec_id", alt.Tooltip("gap:Q", format=".2f")],
            )
            .properties(width=720, height=280,
                        title=f"Specification curve · {curve_item_dd.value.replace('_',' ').title()} · {len(_wide)} specs")
        )
        _zero = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=PALETTE["ink"], strokeWidth=0.8)
            .encode(y="y:Q")
        )
        _median = (
            alt.Chart(pd.DataFrame({"y": [_wide["gap"].median()]}))
            .mark_rule(color=PALETTE["graydark"], strokeDash=[3, 3], strokeWidth=0.8)
            .encode(y="y:Q")
        )

        _curve_out = mo.vstack([
            mo.md(
                f"Of the {len(_wide)} specifications that survive the current filter, **{_pct_expected:.1f}%** place "
                f"the partisan gap on the expected side of zero. The median gap, drawn as the dashed line, sits at "
                f"**{_wide['gap'].median():+.2f}**, and the full range stretches from **{_wide['gap'].min():+.2f}** "
                f"to **{_wide['gap'].max():+.2f}**."
            ),
            _bars + _zero + _median,
        ], align="center")
    _curve_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Coverage of the expected partisan direction is the natural summary statistic for these curves, and on the full multiverse it is high enough to clear inferential thresholds with substantial slack. To anchor that intuition, the next cell builds a permutation null directly from the data: it reshuffles the partisanship labels within each cell and recomputes the coverage statistic many times over, producing the distribution one would expect under no partisan signal at all. The observed coverage is overlaid in red.
    """)
    return


@app.cell
def _(PALETTE, alt, curve_item_dd, df_main, mo, pd):
    import numpy as _np

    _perm_df = df_main[df_main["item"] == curve_item_dd.value].copy()
    _means = (
        _perm_df.groupby(["spec_id", "party"], observed=True)["score"]
        .mean().reset_index()
    )
    _wide = _means.pivot_table(index="spec_id", columns="party", values="score").dropna()
    _observed = ((_wide["Democrat"] - _wide["Republican"]) > 0).mean()

    _rng = _np.random.default_rng(42)
    _n_perm = 2000
    _null = _np.empty(_n_perm)
    _d_vals = _wide["Democrat"].to_numpy()
    _r_vals = _wide["Republican"].to_numpy()
    _stacked = _np.column_stack([_d_vals, _r_vals])
    for _i in range(_n_perm):
        _flips = _rng.integers(0, 2, size=_stacked.shape[0]).astype(bool)
        _swapped = _stacked.copy()
        _swapped[_flips] = _swapped[_flips][:, ::-1]
        _null[_i] = (_swapped[:, 0] - _swapped[:, 1] > 0).mean()

    _null_df = pd.DataFrame({"coverage": _null})
    _alpha05 = _np.quantile(_null, 0.95)
    _p_value = (_null >= _observed).mean()

    _hist = (
        alt.Chart(_null_df)
        .mark_bar(color=PALETTE["graymute"], opacity=0.85)
        .encode(
            x=alt.X("coverage:Q", bin=alt.Bin(maxbins=40),
                    title="Coverage under label-swap null"),
            y=alt.Y("count():Q", title="permutations"),
        )
        .properties(width=560, height=220,
                    title=f"Permutation null · {curve_item_dd.value.replace('_',' ').title()} · {_n_perm} resamples")
    )
    _obs_line = (
        alt.Chart(pd.DataFrame({"x": [_observed]}))
        .mark_rule(color=PALETTE["red"], strokeWidth=1.6)
        .encode(x="x:Q")
    )
    _crit_line = (
        alt.Chart(pd.DataFrame({"x": [_alpha05]}))
        .mark_rule(color=PALETTE["ink"], strokeDash=[3, 3], strokeWidth=0.9)
        .encode(x="x:Q")
    )

    mo.vstack([
        _hist + _crit_line + _obs_line,
        mo.md(
            f"On {curve_item_dd.value.replace('_', ' ')}, the observed coverage of **{_observed:.3f}** "
            f"(red line) sits far above the α=0.05 critical value of **{_alpha05:.3f}** (dashed) and "
            f"corresponds to a permutation p-value of **{_p_value:.4f}**. The null distribution clusters "
            f"around 0.5 as expected: with no partisan information, half of cells would point in the "
            f"expected direction by chance."
        ),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Where does the variation come from?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Coverage tells us whether the signal survives the multiverse; variance decomposition tells us which design choices are doing the actual moving. For each prompt dimension we compute $\eta^2$, the ratio of between-group sum of squares to total sum of squares, on the raw response scores. Because the six dimensions are not orthogonal to each other, the per-item $\eta^2$ values do not sum to one, and treating them as a partition would be misleading. They are best read as a ranking of which lever a researcher would have most leverage on if they wanted to shift the result.
    """)
    return


@app.cell
def _(PALETTE, alt, df_main, mo, pd):
    _DIMS = ["model", "persona_format", "question_framing",
             "system_prompt", "temperature", "few_shot"]

    def _eta_sq(item_df):
        _grand = item_df["score"].mean()
        _ss_total = ((item_df["score"] - _grand) ** 2).sum()
        if _ss_total == 0:
            return {d: 0.0 for d in _DIMS}
        _result = {}
        for _dim in _DIMS:
            _groups = item_df.groupby(_dim, observed=True)["score"]
            _ss_between = sum(len(g) * (g.mean() - _grand) ** 2 for _, g in _groups)
            _result[_dim] = _ss_between / _ss_total
        return _result

    _eta_rows = []
    for _item, _gdf in df_main.groupby("item", observed=True):
        for _dim_name, _val in _eta_sq(_gdf).items():
            _eta_rows.append({"item": _item, "dimension": _dim_name, "eta_sq": _val})
    _eta_df = pd.DataFrame(_eta_rows)
    _eta_df["dominant"] = _eta_df.groupby("item")["eta_sq"].transform(lambda s: s == s.max())

    _eta_chart = (
        alt.Chart(_eta_df)
        .mark_bar()
        .encode(
            x=alt.X("dimension:N", sort=_DIMS, title=None,
                    axis=alt.Axis(labelAngle=-30, labelFontSize=10)),
            y=alt.Y("eta_sq:Q", title="η²"),
            color=alt.Color(
                "dominant:N",
                scale=alt.Scale(domain=[True, False],
                                range=[PALETTE["blue"], PALETTE["graymute"]]),
                legend=None,
            ),
            tooltip=["item", "dimension", alt.Tooltip("eta_sq:Q", format=".3f")],
        )
        .properties(width=260, height=220)
        .facet(column=alt.Column("item:N", title=None))
    )
    mo.vstack([
        mo.md("Variance decomposition by dimension, per item. The dominant lever for each item is highlighted in blue."),
        _eta_chart,
        mo.md(
            "The picture differs sharply across items. For government spending and immigration the dominant lever is "
            "the identity of the model, with the other five axes contributing modestly. Gun control inverts that "
            "ordering: question framing alone accounts for η² of 0.144, more than six times the next-largest "
            "dimension, and a Fisher r-to-z dominance test against persona format returns z = +3.92 with p below "
            "0.0001. That single asymmetry is what makes gun control the most informative item in the multiverse."
        ),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Amplification against ANES 2024
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Coverage and variance address whether and where the multiverse moves, but a separate question is whether its central tendency is well calibrated against real survey responses. We benchmark the mean LLM gap on each item against the corresponding ANES 2024 partisan gap and report the ratio. The pattern that emerges is consistent across items and worth pausing on: silicon-sampled gaps are not noisy approximations of ground truth but systematically larger versions of it, with both partisan poles inflating in tandem so that the gap widens symmetrically.
    """)
    return


@app.cell
def _(PALETTE, alt, df_main, mo, pd):
    _ANES_GAPS = {"gov_spending": 0.86, "immigration": 0.93, "gun_control": 1.18}
    _amp_rows = []
    for _item, _gdf in df_main.groupby("item", observed=True):
        _per_party = _gdf.groupby("party", observed=True)["score"].mean()
        _llm_gap = _per_party["Democrat"] - _per_party["Republican"]
        _amp_rows.append({
            "item": _item.replace("_", " ").title(),
            "source": "ANES 2024",
            "gap": _ANES_GAPS[_item],
            "amp": "",
        })
        _amp_rows.append({
            "item": _item.replace("_", " ").title(),
            "source": "P-SCA mean",
            "gap": round(_llm_gap, 3),
            "amp": f"{_llm_gap / _ANES_GAPS[_item]:.2f}×",
        })
    _amp_df = pd.DataFrame(_amp_rows)
    _amp_chart = (
        alt.Chart(_amp_df)
        .mark_bar()
        .encode(
            x=alt.X("source:N", title=None, axis=alt.Axis(labelFontSize=10, labelAngle=0)),
            y=alt.Y("gap:Q", title="Partisan gap (D minus R)"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["ANES 2024", "P-SCA mean"],
                                range=[PALETTE["graymute"], PALETTE["blue"]]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["item", "source", alt.Tooltip("gap:Q", format=".2f")],
        )
        .properties(width=180, height=220)
        .facet(column=alt.Column("item:N", title=None))
    )
    _labels = _amp_df[_amp_df["amp"] != ""][["item", "amp"]].rename(columns={"amp": "amplification"})
    mo.vstack([
        mo.md("LLM-simulated gaps next to ANES 2024 ground-truth gaps:"),
        _amp_chart,
        mo.ui.table(_labels, selection=None),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The practical implication is uncomfortable. Silicon sampling can be defended as a tool for recovering the *sign* of a partisan disagreement, but using it to estimate *magnitudes* introduces a roughly twofold inflation that no amount of additional prompts seems to wash out. Anyone tempted to read a simulated effect size as a calibrated one should treat that 1.7-to-2.2 range as a haircut to apply before publication.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. The visceral cell: Llama 3.3 70B under Likert
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Averages, however, flatten the most unsettling finding in the dataset. When we condition on a single model and one item, the smooth picture of "amplified but directionally correct" gives way to a configuration in which the partisan sign actually inverts. The next chart pulls the means apart by framing so the inversion is unambiguous.
    """)
    return


@app.cell
def _(PALETTE, alt, df_main, mo, pd):
    _llama_cell = df_main[(df_main["model"] == "llama-3.3-70b") & (df_main["item"] == "gun_control")]
    _rows = []
    for _fr, _sub in _llama_cell.groupby("question_framing", observed=True):
        _per_party = _sub.groupby("party", observed=True)["score"].mean()
        _rows.append({
            "framing": _fr,
            "party": "Democrat",
            "mean response": round(_per_party["Democrat"], 2),
        })
        _rows.append({
            "framing": _fr,
            "party": "Republican",
            "mean response": round(_per_party["Republican"], 2),
        })
    _llama_long = pd.DataFrame(_rows)
    _llama_chart = (
        alt.Chart(_llama_long)
        .mark_bar()
        .encode(
            x=alt.X("party:N", title=None, axis=alt.Axis(labelFontSize=10, labelAngle=0)),
            y=alt.Y("mean response:Q", title="mean response", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color(
                "party:N",
                scale=alt.Scale(domain=["Democrat", "Republican"],
                                range=[PALETTE["blue"], PALETTE["red"]]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["framing", "party", "mean response"],
        )
        .properties(width=180, height=220)
        .facet(column=alt.Column("framing:N", title=None))
    )
    _summary = (
        _llama_long.pivot_table(index="framing", columns="party", values="mean response")
        .reset_index()
    )
    _summary["gap (D minus R)"] = (_summary["Democrat"] - _summary["Republican"]).round(2)
    mo.vstack([
        mo.md("**Llama 3.3 70B on gun control, broken out by question framing:**"),
        _llama_chart,
        mo.ui.table(_summary, selection=None),
        mo.md(
            "Under the direct and forced-choice framings the model behaves as the rest of the multiverse does, "
            "returning healthy gaps of roughly +2.3 and +2.0 respectively. Under the Likert framing the same "
            "model on the same item produces a gap of −1.92, a full inversion of the partisan signal. The "
            "underlying mechanism appears to be scale-anchoring failure on a partially labeled scale rather than "
            "any deep ideological behavior, but a paper that happened to settle on Likert framing for Llama "
            "would have published the wrong direction without warning."
        ),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the multiverse together

    Pulling the threads together, the picture that emerges is neither an indictment of silicon sampling nor an endorsement of it. The partisan signal on these three items survives between roughly eighty-seven and one hundred percent of defensible prompt designs, which is a real finding and one that nobody could establish from a single configuration; arriving at it required surveying the multiverse rather than walking through it. That stability, however, sits alongside two genuine sources of distortion that the practice of single-spec reporting will continue to obscure. Effect sizes are amplified by something close to a factor of two across systems, so anything that hangs on magnitudes rather than directions is, at present, untrustworthy. And occasional cells, of which Llama under Likert is the most striking, manage to flip the sign of the result entirely, which is exactly the kind of pathology that gets concealed when a paper reports a single number and a justification.
    """)
    return


@app.cell(hide_code=True)
def _(PALETTE, mo):
    _card = f"""
    <div style="border:1px solid {PALETTE['graymute']}; border-radius:4px;
                padding:16px 20px; margin:18px auto 0 auto; max-width:680px;
                font-family:Charter, 'Iowan Old Style', Palatino, Georgia, serif;
                color:{PALETTE['ink']}; background:white">
      <div style="font-size:13px; color:{PALETTE['graydark']}; letter-spacing:0.04em;
                  text-transform:uppercase; margin-bottom:8px">
        Cite this work
      </div>
      <div style="font-size:14px; line-height:1.55; margin-bottom:14px">
        Crainic, J., Yee, B., and Koh, P. (2026). <i>Prompt Specification Curve Analysis: How Arbitrary Choices Drive LLM-Simulated Opinion</i>. YCRG-Labs.
      </div>
      <div style="font-size:13px; color:{PALETTE['graydark']}; letter-spacing:0.04em;
                  text-transform:uppercase; margin-bottom:8px">
        Reproduce locally
      </div>
      <pre style="background:{PALETTE['graylight']}; padding:10px 12px; border-radius:3px;
                  font-family:'SF Mono', Menlo, Consolas, monospace; font-size:12.5px;
                  color:{PALETTE['ink']}; margin:0 0 14px 0; overflow-x:auto">pip install psca
psca lhs --n_specs 600 --output my_run.json
psca analyze --output my_run.json --exclude_models gemini-2.5-flash</pre>
      <div style="font-size:13px; color:{PALETTE['graydark']}">
        Source, full CLI, and the data behind every chart on this page live at
        <a href="https://github.com/YCRG-Labs/psca" style="color:{PALETTE['blue']}; text-decoration:none">github.com/YCRG-Labs/psca</a>.
      </div>
    </div>
    """
    mo.Html(_card)
    return


if __name__ == "__main__":
    app.run()
