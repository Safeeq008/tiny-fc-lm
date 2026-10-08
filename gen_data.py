"""Synthetic dataset generator for a single-tool (web_search) function-calling model.

How it works
------------
We define ~10 *intents* (news, weather, price, score, how-to, etc.), each with:

* a pool of entities (people, cities, products, …),
* several paraphrase templates for the user prompt,
* a deterministic gold tool-call string.

For each sample we:
1. pick an intent (or the special "none" / "generic-news" intent),
2. pick a random entity from the intent's pool,
3. fill a random template,
4. optionally apply a random prefix/suffix ("please", "?", "thanks", …),
5. optionally inject a realistic typo (~8% of tokens),
6. pair it with the gold `web_search|query=...|recency=...` or `none`.

Output format (one JSON object per line):
  {"user": "weather in chennai today", "call": "web_search|query=chennai weather today|recency=day"}
  {"user": "hi", "call": "none"}

Splits
------
* **train.jsonl** (80k) — all templates, with prefix/suffix augmentation.
* **val.jsonl** (2k) — same templates but de-duplicated against train (no
  exact user-string overlap).
* **test_ood.jsonl** (1k) — held-out phrasings that NEVER appear in train,
  to test generalisation to unseen surface forms.  "OOD" = out-of-distribution
  *phrasing*, NOT out-of-distribution *entities* (entities are shared).

Limitations (honesty about the data)
-------------------------------------
* The data is **100% synthetic** — real user queries are far more diverse,
  misspelled, multilingual, and ambiguous.
* Entity pools are small and English-centric; the model memorises which
  entities map to which patterns rather than learning general entity copying.
* "none" examples are simple greetings/chit-chat; real non-tool requests
  include complex questions, instructions, and multi-sentence passages that
  this dataset does not cover.
* Recency labels are heuristic (news→day, how-to→any) — real users rarely
  specify recency explicitly, and the mapping is not always correct.

Reproducibility
---------------
All randomness flows through a seeded ``random.Random`` instance, so
``python gen_data.py`` always produces the exact same files.
"""

import json
import random
from config import DATA_DIR

# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------
# Each list is deliberately kept moderate-sized so the model can memorise the
# entities, but large enough to test reasonable coverage.

PEOPLE = [
    "elon musk", "marie curie", "sundar pichai", "serena williams", "alan turing", "satya nadella",
    "taylor swift", "lionel messi", "ada lovelace", "narendra modi", "a r rahman", "virat kohli",
    "jeff bezos", "oprah winfrey", "nikola tesla", "sachin tendulkar", "malala yousafzai", "tim cook",
    "jensen huang", "isaac newton", "steve jobs", "katherine johnson", "rafael nadal", "greta thunberg",
]
COUNTRIES = [
    "india", "the usa", "the uk", "china", "japan", "germany", "france", "canada", "australia",
    "brazil", "russia", "south korea", "italy", "spain", "mexico",
]
TOPICS = [
    "artificial intelligence", "electric cars", "the stock market", "stock market", "climate change",
    "space exploration", "the cricket world cup", "cricket world cup", "bitcoin", "quantum computing",
    "the olympics", "olympics", "inflation", "solar energy", "the chip shortage", "chip shortage",
    "the election", "election", "renewable energy", "cybersecurity", "5g rollout", "the mars mission",
    "mars mission", "football transfers", "interest rates", "gold prices", "electric scooters",
    "self driving cars", "open source software", "the housing market", "housing market", "ai regulation",
    "vaccine research", "smartphone sales", "rocket launches", "the fuel prices", "fuel prices",
    "tech layoffs",
]
CITIES = [
    "chennai", "coimbatore", "london", "tokyo", "new york", "mumbai", "paris", "berlin", "sydney", "toronto",
    "dubai", "singapore", "delhi", "bangalore", "seattle", "chicago", "cairo", "madrid", "rome", "seoul",
    "madurai", "kochi", "hyderabad", "san francisco", "amsterdam", "bangkok",
]
PRODUCTS = [
    "iphone 16", "galaxy s24", "ps5", "macbook air", "rtx 4070", "tesla model 3", "airpods pro",
    "kindle paperwhite", "nintendo switch", "raspberry pi 5", "pixel 9", "ipad mini", "dell xps 13",
    "sony wh 1000xm5", "gopro hero 12", "oneplus 12", "apple watch", "steam deck",
]
COMPANIES = [
    "apple", "tesla", "nvidia", "microsoft", "infosys", "tata motors", "amazon", "google", "reliance",
    "meta", "netflix", "intel", "amd", "wipro", "samsung", "adobe", "uber", "zomato",
]
TASKS = [
    "bake sourdough bread", "reset a router", "learn python", "change a flat tire", "make cold brew coffee",
    "tie a tie", "install docker", "write a resume", "meditate daily", "start a vegetable garden",
    "cook biryani", "fix a leaking tap", "set up a vpn", "learn guitar", "save money monthly",
    "clean a laptop keyboard", "train for a marathon", "create a git branch",
]
TEAMS = [
    "chennai super kings", "real madrid", "lakers", "india cricket team", "manchester united", "barcelona",
    "mumbai indians", "yankees", "arsenal", "golden state warriors", "liverpool", "rcb",
]
CONCEPTS = [
    "photosynthesis", "blockchain", "black holes", "machine learning", "gdp", "dna", "a transformer model",
    "recursion", "the fibonacci sequence", "inflation", "plate tectonics", "quantum entanglement",
    "a neural network", "the greenhouse effect", "cryptography", "entropy",
]

# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------
# Each intent has:
#   pool     — which entity list to draw from
#   t        — training templates (user phrasings)
#   ood      — held-out templates used ONLY in test_ood.jsonl
#   q        — gold search query template (filled with the entity)
#   rec      — gold recency value
INTENTS = [
    dict(pool=TOPICS + PEOPLE + COMPANIES + COUNTRIES, q="{e} news", rec="day",
         t=["latest news about {e}", "what's happening with {e}", "any news on {e}", "{e} news today",
            "update me on {e}", "{e} latest", "news on {e}", "what's new with {e}",
            "recent developments in {e}", "show me the news about {e}", "anything new on {e}", "{e} updates",
            "what's the latest on {e}", "catch me up on {e}", "top stories about {e}", "headlines for {e}",
            "current news about {e}", "is there any news regarding {e}"],
         ood=["give me the current headlines on {e}", "fill me in on the newest happenings around {e}"]),
    dict(pool=PEOPLE, q="{e}", rec="any",
         t=["who is {e}", "tell me about {e}", "who's {e}", "{e} biography", "info on {e}", "who was {e}",
            "give me background on {e}", "what do you know about {e}", "{e} profile", "details about {e}",
            "who exactly is {e}", "i wanna know about {e}", "can you tell me who {e} is", "{e} life story"],
         ood=["i want to learn who {e} is", "share some background about {e}"]),
    dict(pool=CITIES, q="{e} weather today", rec="day",
         t=["weather in {e} today", "what's the weather like in {e}", "is it raining in {e}", "{e} weather",
            "temperature in {e} right now", "how's the weather in {e}", "will it rain in {e} today",
            "weather forecast for {e}", "how cold is it in {e}", "is it sunny in {e}", "current weather {e}",
            "{e} temperature now", "what's it like outside in {e}", "do i need an umbrella in {e}"],
         ood=["how hot is it in {e} today", "give me the climate conditions in {e} right now"]),
    dict(pool=PRODUCTS, q="{e} price", rec="any",
         t=["how much does {e} cost", "price of {e}", "{e} price", "what's the cost of {e}",
            "how much is the {e}", "{e} cost", "how expensive is the {e}", "what does the {e} sell for",
            "where can i see the price of {e}", "how much for a {e}", "{e} how much", "current price of {e}",
            "what would {e} run me", "cost of buying {e}", "what's the damage for a {e}",
            "what's the asking price on a {e}", "price check on {e}"],
         ood=["find me the going rate for {e}", "what's the asking price on a {e}"]),
    dict(pool=TASKS, q="how to {e}", rec="any",
         t=["how to {e}", "how do i {e}", "steps to {e}", "guide to {e}", "help me {e}",
            "tutorial on how to {e}", "how can i {e}", "what's the way to {e}", "teach me to {e}",
            "instructions to {e}", "how would i {e}", "tips to {e}", "easy way to {e}", "how should i {e}"],
         ood=["what is the best way to {e}", "walk me through how to {e}"]),
    dict(pool=TEAMS, q="{e} score", rec="day",
         t=["{e} score", "did {e} win", "latest {e} result", "how did {e} do last night", "{e} match score today",
            "{e} live score", "what was the {e} score", "did {e} lose", "result of the {e} game",
            "{e} final score", "how did {e} play today", "{e} game result", "score for {e}", "who won, {e}"],
         ood=["tell me the outcome of the {e} game", "what happened in the {e} match"]),
    dict(pool=COMPANIES, q="{e} stock price", rec="day",
         t=["{e} stock price", "how is {e} stock doing", "{e} share price today", "what's {e} trading at",
            "{e} stock", "{e} shares", "is {e} stock up", "how are {e} shares doing", "{e} stock today",
            "current {e} share price", "{e} ticker price", "where is {e} stock at", "{e} market price",
            "{e} stock quote"],
         ood=["check the current market value of {e} shares", "how much is one {e} share worth"]),
    dict(pool=CONCEPTS + TOPICS, q="{e}", rec="any",
         t=["what is {e}", "explain {e}", "define {e}", "what does {e} mean", "{e} explained", "what's {e}",
            "meaning of {e}", "{e} definition", "explain to me what {e} is", "what exactly is {e}",
            "{e} in simple terms", "i don't get {e}", "help me understand {e}", "what do people mean by {e}"],
         ood=["can you look up what {e} is about", "give me an overview of {e}"]),
    dict(pool=TOPICS + PEOPLE + CONCEPTS + PRODUCTS, q="{e}", rec="any",
         t=["search the web for {e}", "google {e}", "look up {e}", "search for {e}", "find info about {e}",
            "web search {e}", "search online for {e}", "look {e} up", "find me info on {e}",
            "can you search {e}", "search {e}", "google search for {e}", "browse the web for {e}",
            "find results for {e}"],
         ood=["run a web search on {e}", "dig up some info on {e}"]),
    dict(pool=TOPICS + COMPANIES, q="{e} this week", rec="week",
         t=["{e} this week", "what happened with {e} this week", "weekly roundup on {e}",
            "{e} news from the past week", "recap of {e} over the last 7 days", "{e} weekly summary",
            "{e} in the last week", "this week in {e}", "{e} weekly news", "week's news about {e}",
            "what did {e} do this week", "how was {e} this week", "{e} past 7 days", "{e} over the past week",
            "recap the past seven days for {e}", "past seven days for {e}", "{e} 7 day recap",
            "run me through this week for {e}"],
         ood=["summarize this week's {e} developments", "recap the past seven days for {e}"]),
]

CATEGORIES = [
    "sports", "tech", "business", "entertainment", "world", "science", "health", "gaming",
    "finance", "politics",
]

GENERIC_NEWS = dict(
    pool=CATEGORIES,
    templates=[
        "{e} news", "latest {e} news", "{e} headlines", "what's happening in {e}",
        "{e} updates today", "top {e} stories", "give me {e} news", "any {e} news today",
        "{e} news today", "recent {e} news",
    ],
)

# ---------------------------------------------------------------------------
# "None" (no-tool) examples — chit-chat, greetings, creative requests
# ---------------------------------------------------------------------------
# These teach the model to abstain from calling web_search when the user is
# just chatting.  The OOD set uses phrasings NOT seen in training.
NONE_TRAIN = [
    "hi", "hello", "hello there", "hey", "thanks", "thank you so much", "how are you", "tell me a joke",
    "good morning", "good night", "who are you", "what can you do", "bye", "see you later", "ok cool",
    "nice", "you are awesome", "what's up", "thanks a lot", "that's funny", "i'm tired", "okay", "wow",
    "goodbye", "say something nice", "i am bored", "make me laugh", "what is your name",
    "write a poem about {e}", "write a haiku about {e}", "tell me a story about {e}",
    "i love {e}", "lol", "great job", "can you help me", "never mind", "yes", "no", "sure",
]
NONE_OOD = [
    "you are so helpful", "what should we chat about", "sounds great", "peace out for now",
    "have a great one", "compose a haiku about {e}", "sing me a song about {e}", "make me smile",
    "that made my day", "i appreciate it", "nighty night", "later then", "just saying hey",
    "what do you want to talk about",
]

# ---------------------------------------------------------------------------
# Augmentation: random prefixes and suffixes add naturalness
# ---------------------------------------------------------------------------
PRE = ["", "", "", "please ", "hey, ", "ok so ", "yo ", "quick question: ", "could you ", "i was wondering, "]
SUF = ["", "", "", " please", " thanks", " pls", "?", "?", " asap", " for me"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def typo(s: str, rng: random.Random) -> str:
    """Inject a single realistic typo with ~8% probability.

    Typo types (equally likely):
      - adjacent-character swap  ("weather" → "wetaher")
      - character deletion       ("weather" → "wether")
      - character duplication     ("weather" → "weeather")

    This is intentionally simple — real typo distributions are more complex
    (keyboard-proximity errors, autocorrect artifacts, etc.) but this covers
    the most common class and helps the model be robust to minor misspellings.
    """
    if len(s) < 4 or rng.random() > 0.08:
        return s
    i = rng.randrange(1, len(s) - 1)
    k = rng.random()
    if k < 0.33:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]  # swap
    if k < 0.66:
        return s[:i] + s[i + 1:]                       # delete
    return s[:i] + s[i] + s[i:]                         # duplicate


def fill(tpl: str, e: str, rng: random.Random) -> str:
    """Fill a template with an entity, applying typos to each segment."""
    parts = [typo(p, rng) for p in tpl.split("{e}")]
    return e.join(parts)


def make_generic_news(rng: random.Random) -> dict:
    """Generate a category-level news query (e.g. "sports news")."""
    cat = rng.choice(GENERIC_NEWS["pool"])
    tpl = rng.choice(GENERIC_NEWS["templates"])
    text = tpl.format(e=cat)
    return dict(
        user=rng.choice(PRE) + text + rng.choice(SUF),
        call=f"web_search|query={cat} news|recency=day",
    )


def make(rng: random.Random, split: str) -> dict:
    """Generate a single (user, call) training example.

    Args:
        rng:   seeded Random instance for reproducibility.
        split: one of "train", "val", "ood" — controls which templates are used.

    Returns:
        dict with "user" (str) and "call" (str) keys.
    """
    ood = split == "ood"

    # ~6% chance of a generic-news query (only in train/val)
    if split != "ood" and rng.random() < 0.06:
        return make_generic_news(rng)

    # ~16% chance of a "none" (no-tool) example
    if rng.random() < 0.16:
        tpl = rng.choice(NONE_OOD if ood else NONE_TRAIN)
        text = fill(tpl, rng.choice(TOPICS), rng)
        return dict(user=rng.choice(PRE) + text + rng.choice(SUF), call="none")

    # Otherwise: pick a random intent and generate a tool-call example
    it = rng.choice(INTENTS)
    tpl = rng.choice(it["ood"]) if ood else rng.choice(it["t"])
    e = rng.choice(it["pool"])
    text = fill(tpl, e, rng)
    if not ood:
        text = rng.choice(PRE) + text + rng.choice(SUF)
    call = f"web_search|query={it['q'].format(e=e)}|recency={it['rec']}"
    return dict(user=text.strip(), call=call)


def write(path, rows: list[dict]):
    """Write a list of examples as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(n_train: int = 80_000, n_val: int = 2_000, n_ood: int = 1_000, seed: int = 0):
    """Generate all three data splits and write them to DATA_DIR.

    Args:
        n_train: number of training examples (default 80,000).
        n_val:   number of validation examples, de-duped against train (default 2,000).
        n_ood:   number of OOD test examples with held-out phrasings (default 1,000).
        seed:    random seed for full reproducibility.
    """
    rng = random.Random(seed)
    DATA_DIR.mkdir(exist_ok=True)

    train = [make(rng, "train") for _ in range(n_train)]

    # Validation: sample from the same templates but reject any user string
    # that already appeared in train — prevents leaking exact matches.
    seen = {r["user"] for r in train}
    val = []
    while len(val) < n_val:
        r = make(rng, "val")
        if r["user"] not in seen:
            val.append(r)

    # OOD: uses held-out templates only → tests generalisation to unseen
    # phrasings, not unseen entities.
    ood = [make(rng, "ood") for _ in range(n_ood)]

    write(DATA_DIR / "train.jsonl", train)
    write(DATA_DIR / "val.jsonl", val)
    write(DATA_DIR / "test_ood.jsonl", ood)

    print(f"Generated data splits:")
    print(f"  train     = {len(train):,} samples → {DATA_DIR / 'train.jsonl'}")
    print(f"  val       = {len(val):,} samples → {DATA_DIR / 'val.jsonl'}")
    print(f"  test_ood  = {len(ood):,} samples → {DATA_DIR / 'test_ood.jsonl'}")
    print(f"\nSample training examples:")
    for r in train[:5]:
        print(f"  {r}")


if __name__ == "__main__":
    main()
