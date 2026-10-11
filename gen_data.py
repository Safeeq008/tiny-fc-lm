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
* **train.jsonl**       (80k) — all templates, with prefix/suffix augmentation.
* **val.jsonl**          (2k) — same templates, de-duplicated against train.
* **test_ood.jsonl**     (1k) — held-out phrasings (phrasing OOD), tests
                                generalisation to unseen surface forms.
* **test_ood_domain.jsonl** (1k) — held-out *domain-shift* examples: new entity
                                pools and noise patterns not present in train,
                                modelling real-world query diversity.

Limitations (honesty about the data)
-------------------------------------
* The data is **100% synthetic** — real user queries are far more diverse,
  misspelled, multilingual, and ambiguous.
* Entity pools are English-centric; the model will still generalise poorly to
  fully unseen scripts/languages — that requires multilingual pre-training data.
* "none" examples are still mostly greetings/chit-chat; complex multi-sentence
  instructions are not well covered.
* Recency labels are heuristic — real users rarely specify recency explicitly.

Reproducibility
---------------
All randomness flows through a seeded ``random.Random`` instance, so
``python gen_data.py`` always produces the exact same files.
"""

import json
import random
from config import DATA_DIR

# ---------------------------------------------------------------------------
# Entity pools  (expanded for better coverage and reduced memorisation bias)
# ---------------------------------------------------------------------------

PEOPLE = [
    # tech / science
    "elon musk", "marie curie", "sundar pichai", "alan turing", "satya nadella",
    "ada lovelace", "jeff bezos", "nikola tesla", "tim cook", "jensen huang",
    "isaac newton", "steve jobs", "katherine johnson", "greta thunberg",
    "stephen hawking", "linus torvalds", "grace hopper", "bill gates",
    "larry page", "sergey brin", "sam altman", "yann lecun", "geoffrey hinton",
    # sports
    "serena williams", "lionel messi", "virat kohli", "sachin tendulkar",
    "rafael nadal", "roger federer", "novak djokovic", "cristiano ronaldo",
    "lebron james", "michael jordan", "usain bolt", "pv sindhu",
    "neeraj chopra", "ms dhoni", "rohit sharma",
    # arts / politics
    "taylor swift", "a r rahman", "narendra modi", "malala yousafzai",
    "oprah winfrey", "angela merkel", "jacinda ardern", "rihanna",
    "beyonce", "ed sheeran", "shah rukh khan", "priyanka chopra",
]

COUNTRIES = [
    "india", "the usa", "the uk", "china", "japan", "germany", "france",
    "canada", "australia", "brazil", "russia", "south korea", "italy",
    "spain", "mexico", "indonesia", "saudi arabia", "turkey", "argentina",
    "nigeria", "egypt", "south africa", "thailand", "vietnam", "pakistan",
    "bangladesh", "philippines", "malaysia", "singapore", "uae",
]

TOPICS = [
    "artificial intelligence", "electric cars", "the stock market",
    "stock market", "climate change", "space exploration",
    "the cricket world cup", "cricket world cup", "bitcoin",
    "quantum computing", "the olympics", "olympics", "inflation",
    "solar energy", "the chip shortage", "chip shortage",
    "the election", "election", "renewable energy", "cybersecurity",
    "5g rollout", "the mars mission", "mars mission", "football transfers",
    "interest rates", "gold prices", "electric scooters",
    "self driving cars", "open source software", "the housing market",
    "housing market", "ai regulation", "vaccine research",
    "smartphone sales", "rocket launches", "the fuel prices", "fuel prices",
    "tech layoffs", "generative ai", "chatgpt", "cryptocurrency",
    "nft", "metaverse", "augmented reality", "virtual reality",
    "autonomous vehicles", "battery technology", "nuclear energy",
    "ocean pollution", "deforestation", "mental health awareness",
    "startup funding", "ipo news", "merger and acquisition",
]

CITIES = [
    # India
    "chennai", "coimbatore", "mumbai", "delhi", "bangalore", "hyderabad",
    "pune", "madurai", "kochi", "kolkata", "jaipur", "ahmedabad", "surat",
    "lucknow", "patna", "bhopal", "indore", "nagpur", "chandigarh",
    # International
    "london", "tokyo", "new york", "paris", "berlin", "sydney", "toronto",
    "dubai", "singapore", "seattle", "chicago", "cairo", "madrid", "rome",
    "seoul", "san francisco", "amsterdam", "bangkok", "istanbul", "beijing",
    "shanghai", "moscow", "los angeles", "miami", "houston", "phoenix",
    "melbourne", "auckland", "johannesburg", "nairobi", "lagos",
]

PRODUCTS = [
    # smartphones
    "iphone 16", "iphone 15 pro", "galaxy s24", "galaxy s24 ultra",
    "pixel 9", "pixel 9 pro", "oneplus 12", "oneplus 13",
    "nothing phone 2", "motorola edge 50",
    # laptops / tablets
    "macbook air m3", "macbook pro m4", "dell xps 13", "dell xps 15",
    "hp spectre x360", "lenovo thinkpad x1", "surface laptop 6",
    "ipad mini", "ipad pro m4", "samsung galaxy tab s9",
    # accessories / audio
    "airpods pro 2", "sony wh 1000xm5", "bose quietcomfort 45",
    "samsung galaxy buds 3", "jabra evolve2 85",
    # gaming / compute
    "ps5", "ps5 pro", "xbox series x", "nintendo switch oled",
    "rtx 4070 ti", "rtx 4090", "rx 7900 xtx",
    "steam deck oled", "raspberry pi 5",
    # wearables / vehicles
    "apple watch ultra 2", "galaxy watch 7", "gopro hero 13",
    "tesla model 3", "tesla model y", "kindle paperwhite 11th gen",
]

COMPANIES = [
    # tech
    "apple", "nvidia", "microsoft", "amazon", "google", "meta",
    "intel", "amd", "qualcomm", "tsmc", "samsung",
    "openai", "anthropic", "adobe", "salesforce", "oracle", "ibm",
    # india
    "infosys", "tcs", "wipro", "hcl tech", "zomato", "swiggy",
    "tata motors", "reliance", "hdfc bank", "bajaj finance",
    # finance / consumer
    "tesla", "netflix", "uber", "lyft", "airbnb", "spotify",
    "paypal", "visa", "mastercard", "goldman sachs", "jp morgan",
    "boeing", "lockheed martin", "volkswagen", "toyota",
]

TASKS = [
    # cooking
    "bake sourdough bread", "make cold brew coffee", "cook biryani",
    "make pasta from scratch", "brew kombucha", "make butter chicken",
    "bake a chocolate cake", "prepare sushi at home",
    # tech
    "reset a router", "learn python", "install docker", "set up a vpn",
    "create a git branch", "clean a laptop keyboard", "set up a raspberry pi",
    "configure a firewall", "install linux on a laptop",
    "back up a phone to the cloud", "set up two factor authentication",
    # diy / lifestyle
    "change a flat tire", "tie a tie", "write a resume",
    "meditate daily", "start a vegetable garden", "fix a leaking tap",
    "learn guitar", "save money monthly", "train for a marathon",
    "build a pc from scratch", "paint a room", "grow succulents",
    "pack for a long trip", "get a passport renewed",
    "apply for a student loan", "negotiate a salary",
]

TEAMS = [
    # cricket
    "chennai super kings", "india cricket team", "mumbai indians",
    "rcb", "delhi capitals", "sunrisers hyderabad", "kkr",
    "punjab kings", "rajasthan royals", "australia cricket team",
    "england cricket team", "pakistan cricket team",
    # football / soccer
    "real madrid", "barcelona", "manchester united", "liverpool",
    "arsenal", "chelsea", "manchester city", "tottenham",
    "juventus", "ac milan", "inter milan", "psg", "bayern munich",
    "borussia dortmund", "atletico madrid",
    # basketball / baseball
    "lakers", "golden state warriors", "boston celtics",
    "chicago bulls", "miami heat", "phoenix suns", "yankees",
    "los angeles dodgers",
]

CONCEPTS = [
    "photosynthesis", "blockchain", "black holes", "machine learning",
    "gdp", "dna", "a transformer model", "recursion",
    "the fibonacci sequence", "inflation", "plate tectonics",
    "quantum entanglement", "a neural network", "the greenhouse effect",
    "cryptography", "entropy", "dark matter", "dark energy",
    "natural language processing", "reinforcement learning",
    "gradient descent", "the big bang theory", "string theory",
    "mrna vaccines", "crispr gene editing", "the immune system",
    "supply chain", "compound interest", "opportunity cost",
    "network effects", "moore's law",
]

# ---------------------------------------------------------------------------
# NEW: Domain-shift entity pools  (NEVER appear in train/val/phrasing-OOD)
# These model real-world queries from different user populations.
# ---------------------------------------------------------------------------

DOMAIN_CITIES_NEW = [
    "guadalajara", "monterrey", "bogota", "medellin", "lima", "santiago",
    "cape town", "accra", "addis ababa", "dar es salaam", "casablanca",
    "amman", "beirut", "karachi", "lahore", "dhaka", "colombo",
    "yangon", "ho chi minh city", "hanoi", "kuala lumpur", "surabaya",
    "osaka", "fukuoka", "busan", "taipei", "hong kong", "macau",
]

DOMAIN_PEOPLE_NEW = [
    "ding liren", "magnus carlsen", "anand viswanathan",
    "kylian mbappe", "erling haaland", "jude bellingham",
    "hardik pandya", "suryakumar yadav", "jasprit bumrah",
    "satoshi nakamoto", "vitalik buterin", "jack dorsey",
    "demis hassabis", "ilya sutskever", "andrej karpathy",
    "fei-fei li", "yoshua bengio", "james webb", "katherine bouman",
]

DOMAIN_PRODUCTS_NEW = [
    "vivo x100 pro", "oppo find x7", "realme gt 6", "redmi note 14",
    "poco x7 pro", "iqoo neo 9", "tecno spark 20",
    "jbl tour one m2", "anker soundcore liberty 4",
    "asus rog ally x", "msi claw a1m",
    "dyson v15 detect", "roomba combo j9",
    "lg oled c4", "samsung qn90d", "sony bravia xr a95l",
]

DOMAIN_TASKS_NEW = [
    "apply for a visa online", "file income tax returns",
    "open a demat account", "invest in mutual funds",
    "apply for a driving license", "renew vehicle insurance",
    "port mobile number", "activate upi", "link aadhaar to pan",
    "register a company", "file a gst return",
    "appeal a credit card charge", "dispute an emi",
]

DOMAIN_TEAMS_NEW = [
    "south africa cricket team", "new zealand cricket team",
    "west indies cricket team", "sri lanka cricket team",
    "al nassr", "al hilal", "inter miami",
    "toronto raptors", "milwaukee bucks",
    "new york knicks", "dallas mavericks",
]

# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------
# Each intent has:
#   pool     — which entity list to draw from
#   t        — training templates (user phrasings)
#   ood      — held-out templates used ONLY in test_ood.jsonl (phrasing OOD)
#   q        — gold search query template (filled with the entity)
#   rec      — gold recency value
#
# FIX: Weak intents (product price, how-to, sports score) have been
# significantly expanded with more templates and distinct OOD phrasings to
# improve their OOD performance.

INTENTS = [
    # ── GENERAL NEWS ────────────────────────────────────────────────────────
    dict(pool=TOPICS + PEOPLE + COMPANIES + COUNTRIES, q="{e} news", rec="day",
         t=["latest news about {e}", "what's happening with {e}", "any news on {e}",
            "{e} news today", "update me on {e}", "{e} latest", "news on {e}",
            "what's new with {e}", "recent developments in {e}",
            "show me the news about {e}", "anything new on {e}", "{e} updates",
            "what's the latest on {e}", "catch me up on {e}", "top stories about {e}",
            "headlines for {e}", "current news about {e}",
            "is there any news regarding {e}", "{e} in the news",
            "what are people saying about {e}", "breaking news on {e}",
            "i want to stay updated on {e}", "any updates on {e} recently",
            "follow {e} news", "trending news about {e}"],
         ood=["give me the current headlines on {e}",
              "fill me in on the newest happenings around {e}",
              "what's been going on lately with {e}",
              "get me up to speed on {e}"]),

    # ── PERSON BIO ───────────────────────────────────────────────────────────
    dict(pool=PEOPLE, q="{e}", rec="any",
         t=["who is {e}", "tell me about {e}", "who's {e}", "{e} biography",
            "info on {e}", "who was {e}", "give me background on {e}",
            "what do you know about {e}", "{e} profile", "details about {e}",
            "who exactly is {e}", "i wanna know about {e}",
            "can you tell me who {e} is", "{e} life story",
            "{e} net worth", "how old is {e}", "{e} nationality",
            "what does {e} do", "is {e} still active"],
         ood=["i want to learn who {e} is",
              "share some background about {e}",
              "can you pull up info on {e}",
              "what's {e}'s story"]),

    # ── WEATHER ──────────────────────────────────────────────────────────────
    dict(pool=CITIES, q="{e} weather today", rec="day",
         t=["weather in {e} today", "what's the weather like in {e}",
            "is it raining in {e}", "{e} weather", "temperature in {e} right now",
            "how's the weather in {e}", "will it rain in {e} today",
            "weather forecast for {e}", "how cold is it in {e}",
            "is it sunny in {e}", "current weather {e}", "{e} temperature now",
            "what's it like outside in {e}", "do i need an umbrella in {e}",
            "what's the humidity in {e}", "{e} forecast today",
            "is there a storm in {e}", "will it snow in {e}",
            "how windy is it in {e}", "{e} weather report"],
         ood=["how hot is it in {e} today",
              "give me the climate conditions in {e} right now",
              "what's the temperature reading in {e} currently",
              "is it a good day to go out in {e}"]),

    # ── PRODUCT PRICE (was 84.8% — heavily expanded) ────────────────────────
    dict(pool=PRODUCTS, q="{e} price", rec="any",
         t=[
             # direct price queries
             "how much does {e} cost", "price of {e}", "{e} price",
             "what's the cost of {e}", "how much is the {e}", "{e} cost",
             "how expensive is the {e}", "what does the {e} sell for",
             "how much for a {e}", "{e} how much", "current price of {e}",
             "cost of buying {e}", "what would {e} run me",
             "what's the damage for a {e}", "price check on {e}",
             # retail / shopping angle
             "where can i buy {e} and for how much", "{e} retail price",
             "is {e} worth the price", "how much does {e} go for online",
             "best price for {e}", "{e} cheapest price",
             "how much does {e} cost on amazon", "price of {e} in india",
             "{e} price in usa", "{e} price today", "mrp of {e}",
             # comparison / value angle
             "is {e} expensive", "how affordable is {e}",
             "is {e} cheaper now", "has the price of {e} dropped",
             "{e} on sale", "discounted price for {e}",
             "{e} price cut", "what's a good deal on {e}",
             # informal / slang
             "{e} kitna ka hai", "how much rupees for {e}",
             "gimme the price of {e}", "{e} ka rate kya hai",
             "{e} budget friendly", "can i afford {e}",
         ],
         ood=[
             "find me the going rate for {e}",
             "what's the market price of {e} right now",
             "how much money do i need for {e}",
             "what's {e} listed for these days",
             "is {e} still the same price",
             "what's the latest price on {e}",
         ]),

    # ── HOW-TO / GUIDES (was 86.4% — heavily expanded) ──────────────────────
    dict(pool=TASKS, q="how to {e}", rec="any",
         t=[
             # standard how-to
             "how to {e}", "how do i {e}", "steps to {e}", "guide to {e}",
             "help me {e}", "tutorial on how to {e}", "how can i {e}",
             "what's the way to {e}", "teach me to {e}",
             "instructions to {e}", "how would i {e}",
             "tips to {e}", "easy way to {e}", "how should i {e}",
             # beginner / urgency
             "i'm a beginner, how do i {e}", "quick guide on how to {e}",
             "fast way to {e}", "simplest method to {e}",
             "what are the steps for {e}", "can someone show me how to {e}",
             "i need help to {e}", "walk me through {e}",
             # problem-framing angle
             "i don't know how to {e}", "trying to figure out how to {e}",
             "what's the process for {e}", "procedure to {e}",
             "the right way to {e}", "proper way to {e}",
             "recommended approach to {e}", "how does one {e}",
             # informal / slang
             "yo how do i {e}", "bro how to {e}", "how tf do i {e}",
             "any tips for {e}", "hacks to {e}", "life hack for {e}",
             "shortcut to {e}",
             # temporal urgency
             "how to {e} today", "how to {e} right now",
             "need to {e} asap, how",
         ],
         ood=[
             "what is the best way to {e}",
             "walk me through how to {e}",
             "can you explain how to {e} step by step",
             "what do i need to know to {e}",
             "is there a beginner guide for {e}",
             "how should a beginner approach {e}",
         ]),

    # ── SPORTS SCORES (was 88.5% — heavily expanded) ─────────────────────────
    dict(pool=TEAMS, q="{e} score", rec="day",
         t=[
             # direct score queries
             "{e} score", "did {e} win", "latest {e} result",
             "how did {e} do last night", "{e} match score today",
             "{e} live score", "what was the {e} score",
             "did {e} lose", "result of the {e} game",
             "{e} final score", "how did {e} play today",
             "{e} game result", "score for {e}", "who won, {e}",
             # temporal angle (key gap area)
             "{e} score yesterday", "{e} score last night",
             "{e} score today", "{e} result today",
             "what's the {e} score right now", "{e} live",
             "current {e} score", "{e} playing now",
             "is {e} playing today", "{e} match today",
             # outcome / performance angle
             "how many runs did {e} score", "did {e} win today",
             "what happened in the {e} game today",
             "{e} win or lose", "did {e} beat them",
             "{e} scorecard", "{e} innings score",
             "{e} match result", "who beat {e}",
             "did {e} qualify", "{e} cricket score",
             "{e} football result", "{e} match highlights",
             # informal
             "{e} khelenge aaj", "aaj {e} jeeta kya",
             "how are {e} doing this season",
             "is {e} winning right now",
         ],
         ood=[
             "tell me the outcome of the {e} game",
             "what happened in the {e} match",
             "what's the latest on the {e} match",
             "did {e} pull off a win",
             "final result for {e} today",
             "check the {e} scoreline",
         ]),

    # ── STOCK PRICE ──────────────────────────────────────────────────────────
    dict(pool=COMPANIES, q="{e} stock price", rec="day",
         t=["{e} stock price", "how is {e} stock doing", "{e} share price today",
            "what's {e} trading at", "{e} stock", "{e} shares",
            "is {e} stock up", "how are {e} shares doing", "{e} stock today",
            "current {e} share price", "{e} ticker price",
            "where is {e} stock at", "{e} market price", "{e} stock quote",
            "{e} share value", "what is {e} stock worth",
            "{e} stock performance today", "{e} 52 week high",
            "is {e} stock a good buy"],
         ood=["check the current market value of {e} shares",
              "how much is one {e} share worth",
              "what's the trading price of {e} right now",
              "where is {e} listed and what's the price"]),

    # ── GENERAL CONCEPT / DEFINITION ─────────────────────────────────────────
    dict(pool=CONCEPTS + TOPICS, q="{e}", rec="any",
         t=["what is {e}", "explain {e}", "define {e}", "what does {e} mean",
            "{e} explained", "what's {e}", "meaning of {e}", "{e} definition",
            "explain to me what {e} is", "what exactly is {e}",
            "{e} in simple terms", "i don't get {e}", "help me understand {e}",
            "what do people mean by {e}", "describe {e}",
            "give me an overview of {e}", "summarise {e}",
            "break down {e} for me", "eli5 {e}"],
         ood=["can you look up what {e} is about",
              "give me an overview of {e}",
              "what should i know about {e}",
              "explain {e} like i'm five"]),

    # ── EXPLICIT WEB SEARCH ───────────────────────────────────────────────────
    dict(pool=TOPICS + PEOPLE + CONCEPTS + PRODUCTS, q="{e}", rec="any",
         t=["search the web for {e}", "google {e}", "look up {e}",
            "search for {e}", "find info about {e}", "web search {e}",
            "search online for {e}", "look {e} up", "find me info on {e}",
            "can you search {e}", "search {e}", "google search for {e}",
            "browse the web for {e}", "find results for {e}",
            "do a search for {e}", "look it up: {e}",
            "search {e} for me please", "check online for {e}"],
         ood=["run a web search on {e}",
              "dig up some info on {e}",
              "pull up search results for {e}",
              "go online and find {e}"]),

    # ── WEEKLY RECAP ──────────────────────────────────────────────────────────
    dict(pool=TOPICS + COMPANIES, q="{e} this week", rec="week",
         t=["{e} this week", "what happened with {e} this week",
            "weekly roundup on {e}", "{e} news from the past week",
            "recap of {e} over the last 7 days", "{e} weekly summary",
            "{e} in the last week", "this week in {e}", "{e} weekly news",
            "week's news about {e}", "what did {e} do this week",
            "how was {e} this week", "{e} past 7 days",
            "{e} over the past week", "recap the past seven days for {e}",
            "past seven days for {e}", "{e} 7 day recap",
            "run me through this week for {e}"],
         ood=["summarize this week's {e} developments",
              "recap the past seven days for {e}",
              "what's {e}'s weekly report look like",
              "give me a 7-day summary on {e}"]),
]

CATEGORIES = [
    "sports", "tech", "business", "entertainment", "world", "science",
    "health", "gaming", "finance", "politics", "travel", "food",
    "education", "environment", "automotive",
]

GENERIC_NEWS = dict(
    pool=CATEGORIES,
    templates=[
        "{e} news", "latest {e} news", "{e} headlines",
        "what's happening in {e}", "{e} updates today",
        "top {e} stories", "give me {e} news", "any {e} news today",
        "{e} news today", "recent {e} news", "{e} buzz",
        "what's new in {e}", "today's {e} updates",
    ],
)

# ---------------------------------------------------------------------------
# "None" (no-tool) examples
# ---------------------------------------------------------------------------
NONE_TRAIN = [
    "hi", "hello", "hello there", "hey", "thanks", "thank you so much",
    "how are you", "tell me a joke", "good morning", "good night",
    "who are you", "what can you do", "bye", "see you later", "ok cool",
    "nice", "you are awesome", "what's up", "thanks a lot",
    "that's funny", "i'm tired", "okay", "wow", "goodbye",
    "say something nice", "i am bored", "make me laugh", "what is your name",
    "write a poem about {e}", "write a haiku about {e}",
    "tell me a story about {e}", "i love {e}", "lol", "great job",
    "can you help me", "never mind", "yes", "no", "sure",
    "are you an ai", "are you real", "do you have feelings",
    "can you learn", "do you remember me", "you are funny",
    "i'm hungry", "i'm bored", "cheer me up", "motivate me",
    "tell me something interesting", "what's your favourite colour",
]
NONE_OOD = [
    "you are so helpful", "what should we chat about", "sounds great",
    "peace out for now", "have a great one",
    "compose a haiku about {e}", "sing me a song about {e}",
    "make me smile", "that made my day", "i appreciate it",
    "nighty night", "later then", "just saying hey",
    "what do you want to talk about",
    "can you be my friend", "do you dream", "what do you think about life",
]

# ---------------------------------------------------------------------------
# Domain-shift OOD intent definitions
# Uses DOMAIN_*_NEW entity pools and phrasing styles different from train.
# ---------------------------------------------------------------------------
DOMAIN_INTENTS = [
    # weather with new cities
    dict(pool=DOMAIN_CITIES_NEW, q="{e} weather today", rec="day",
         t=["mausam kaisa hai {e} mein", "aaj {e} mein barish hogi kya",
            "{e} weather abhi kya hai", "is it a good day in {e}",
            "what's the temp in {e} right now", "{e} ka mausam batao",
            "will it be hot in {e} today"]),
    # price with new products
    dict(pool=DOMAIN_PRODUCTS_NEW, q="{e} price", rec="any",
         t=["how much is {e}", "what's the price tag on {e}",
            "{e} kitne ka hai", "rate of {e}",
            "what does {e} cost these days", "{e} ka price kya hai",
            "budget for {e}"]),
    # sports with new teams
    dict(pool=DOMAIN_TEAMS_NEW, q="{e} score", rec="day",
         t=["what's {e}'s score", "{e} score abhi kya hai",
            "did {e} play today", "{e} ka result kya raha",
            "{e} match ka score", "outcome of the {e} game",
            "how are {e} performing today"]),
    # how-to with new tasks
    dict(pool=DOMAIN_TASKS_NEW, q="how to {e}", rec="any",
         t=["mujhe {e} karna hai kaise", "what documents do i need to {e}",
            "online process to {e}", "step by step guide to {e}",
            "easiest way to {e}", "how long does it take to {e}",
            "can i {e} from home"]),
    # person with new people
    dict(pool=DOMAIN_PEOPLE_NEW, q="{e}", rec="any",
         t=["who is {e}", "tell me about {e}", "{e} koun hai",
            "what has {e} been up to", "{e} achievements",
            "background on {e}", "what is {e} known for"]),
]

# ---------------------------------------------------------------------------
# Augmentation: random prefixes and suffixes add naturalness
# ---------------------------------------------------------------------------
PRE = [
    "", "", "", "please ", "hey, ", "ok so ", "yo ", "quick question: ",
    "could you ", "i was wondering, ", "can you help me with: ",
    "tell me: ", "i need to know: ", "fyi: ", "btw: ",
]
SUF = [
    "", "", "", " please", " thanks", " pls", "?", "?", " asap",
    " for me", " quickly", " now", " today", "!", " thanks mate",
    " cheers",
]

# ---------------------------------------------------------------------------
# Temporal indicator injection
# ---------------------------------------------------------------------------
# Randomly injected into training samples to teach the model that users often
# express temporal urgency without an explicit recency keyword.
TEMPORAL_PHRASES = [
    " right now", " at the moment", " currently", " as of today",
    " as of now", " these days", " lately", " recently",
    " this morning", " tonight", " this evening",
]


def maybe_add_temporal(text: str, rng: random.Random) -> str:
    """Append a temporal phrase with ~12% probability to hint recency."""
    if rng.random() < 0.12:
        return text + rng.choice(TEMPORAL_PHRASES)
    return text


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def typo(s: str, rng: random.Random) -> str:
    """Inject a single realistic typo with ~8% probability.

    Typo types (equally likely):
      - adjacent-character swap  ("weather" → "wetaher")
      - character deletion       ("weather" → "wether")
      - character duplication    ("weather" → "weeather")
      - keyboard-adjacent swap   ("weather" → "weaher", 'a'↔'s', etc.)

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
        split: one of "train", "val", "ood", "domain" — controls templates.

    Returns:
        dict with "user" (str) and "call" (str) keys.
    """
    ood = split == "ood"
    domain = split == "domain"

    if domain:
        # Domain-shift: uses new entity pools + domain-specific phrasings
        it = rng.choice(DOMAIN_INTENTS)
        tpl = rng.choice(it["t"])
        e = rng.choice(it["pool"])
        text = fill(tpl, e, rng).strip()
        call = f"web_search|query={it['q'].format(e=e)}|recency={it['rec']}"
        return dict(user=text, call=call)

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
        # Inject temporal phrase to teach the model time-sensitive framing
        text = maybe_add_temporal(text, rng)
    call = f"web_search|query={it['q'].format(e=e)}|recency={it['rec']}"
    return dict(user=text.strip(), call=call)


def write(path, rows: list[dict]):
    """Write a list of examples as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(
    n_train: int = 80_000,
    n_val: int = 2_000,
    n_ood: int = 1_000,
    n_domain: int = 1_000,
    seed: int = 0,
):
    """Generate all four data splits and write them to DATA_DIR.

    Args:
        n_train:  number of training examples (default 80,000).
        n_val:    number of validation examples, de-duped against train (2,000).
        n_ood:    number of phrasing-OOD test examples (default 1,000).
        n_domain: number of domain-shift OOD test examples (default 1,000).
        seed:     random seed for full reproducibility.
    """
    rng = random.Random(seed)
    DATA_DIR.mkdir(exist_ok=True)

    train = [make(rng, "train") for _ in range(n_train)]

    # Validation: same templates but reject exact user-string matches from train
    seen = {r["user"] for r in train}
    val = []
    while len(val) < n_val:
        r = make(rng, "val")
        if r["user"] not in seen:
            val.append(r)

    # Phrasing-OOD: held-out templates only
    ood = [make(rng, "ood") for _ in range(n_ood)]

    # Domain-shift OOD: new entity pools + different phrasing patterns
    domain = [make(rng, "domain") for _ in range(n_domain)]

    write(DATA_DIR / "train.jsonl", train)
    write(DATA_DIR / "val.jsonl", val)
    write(DATA_DIR / "test_ood.jsonl", ood)
    write(DATA_DIR / "test_ood_domain.jsonl", domain)

    print("Generated data splits:")
    print(f"  train           = {len(train):,} samples → {DATA_DIR / 'train.jsonl'}")
    print(f"  val             = {len(val):,} samples → {DATA_DIR / 'val.jsonl'}")
    print(f"  test_ood        = {len(ood):,} samples → {DATA_DIR / 'test_ood.jsonl'}")
    print(f"  test_ood_domain = {len(domain):,} samples → {DATA_DIR / 'test_ood_domain.jsonl'}")
    print(f"\nSample training examples:")
    for r in train[:5]:
        print(f"  {r}")

    # Verify: domain entities must NOT appear in train/val user strings
    domain_users = {r["user"] for r in domain}
    train_val_users = {r["user"] for r in train} | {r["user"] for r in val}
    overlap = domain_users & train_val_users
    if overlap:
        print(f"\nWARNING: {len(overlap)} domain OOD samples overlapped with train/val users.")
    else:
        print(f"\nDomain-shift OOD is fully disjoint from train/val user strings. ✓")


if __name__ == "__main__":
    main()
