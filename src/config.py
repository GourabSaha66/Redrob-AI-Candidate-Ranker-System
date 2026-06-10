BIENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_BIENCODER = 500
TOP_K_CROSSENCODER = 200
FINAL_TOP_N = 100

WEIGHT_ROLE_FIT     = 0.38
WEIGHT_CAREER_ARC   = 0.27
WEIGHT_AVAILABILITY = 0.20
WEIGHT_LOCATION     = 0.15

BIENC_WEIGHT    = 0.30
CROSSENC_WEIGHT = 0.70

HONEYPOT_ASSESSMENT_THRESHOLD = 35
MIN_YEARS_EXPERIENCE = 3.5

PURE_SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "wipro", "infosys", "accenture", "cognizant",
    "capgemini", "hcl", "hcl technologies", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "mindtree", "syntel",
    "niit technologies", "mastech", "igate", "patni", "zensar",
    "birlasoft", "cyient", "kpit", "persistent", "sasken"
}

TIER1_INDIA_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bengaluru", "bangalore",
    "delhi", "gurugram", "gurgaon", "chennai", "kolkata", "ahmedabad", "india"
}

# Skills the JD says are absolutely required — weighted heavily
MUST_HAVE_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss", "opensearch", "elasticsearch", "hybrid search",
    "retrieval", "information retrieval", "rag", "reranking", "ranking",
    "semantic search", "ndcg", "mrr", "evaluation", "a/b test",
    "learning to rank", "xgboost", "lightgbm", "hugging face transformers",
    "recommendation", "recommendation systems", "search"
}

# Skills that indicate wrong specialization for this role
RED_FLAG_SKILLS = {
    "image classification", "object detection", "gans", "computer vision",
    "speech recognition", "tts", "text to speech", "asr",
    "robotics", "slam", "point cloud", "lidar"
}

# High-signal product-company industries for this role
PRODUCT_INDUSTRIES = {
    "food delivery", "e-commerce", "fintech", "edtech", "healthtech",
    "saas", "ai/ml", "transportation", "media", "gaming", "travel",
    "real estate tech", "proptech", "insurtech", "legaltech",
}

JD_TECHNICAL_REQUIREMENTS = """
Senior AI Engineer with production experience building embeddings-based retrieval systems.
Must have sentence-transformers, OpenAI embeddings, BGE, E5 or similar deployed to real users.
Production experience with vector databases: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, FAISS. Strong Python. Evaluation frameworks: NDCG, MRR, MAP, A/B testing.
Hybrid search infrastructure. LLM fine-tuning LoRA QLoRA PEFT. Learning-to-rank XGBoost LightGBM.
Information retrieval recommendation systems ranking models feature engineering production deployment.
"""

JD_CAREER_NARRATIVE = """
Applied ML engineer who shipped end-to-end ranking search recommendation system to real users at scale.
5 to 9 years experience 4 to 5 years applied ML at product companies not pure services.
Scrappy product engineering. Deep technical depth embeddings retrieval ranking LLMs fine-tuning.
Handled embedding drift index refresh retrieval quality regression in production.
Set up evaluation infrastructure offline benchmarks online A/B testing feedback loops.
Will write production code. Located Pune Noida or willing to relocate. Series A founding team.
XGBoost LightGBM learning to rank NDCG MRR offline evaluation semantic search vector database.
"""
