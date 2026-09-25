
import re
import html
import unicodedata
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RailAssist - Railway Complaint Management",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
    }
    .priority-high {
        color: #b91c1c;
        font-weight: 700;
    }
    .priority-standard {
        color: #92400e;
        font-weight: 700;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = "train.csv"

# Descriptive mapping used by the current prototype.
# These are application-level interpretations of the numeric labels,
# not official railway category definitions.
CATEGORY_MAPPING = {
    0: "Coach Facilities / Cleanliness",
    1: "Security / Lost Property",
    2: "Medical Assistance",
    3: "Catering / Vendors",
    4: "Train Delay / Status",
    5: "Ticketing / Reservation",
    6: "Other / General Service",
    7: "Administrative Follow-up",
}

CATEGORY_ID_FROM_NAME = {v: k for k, v in CATEGORY_MAPPING.items()}

SAMPLE_COMPLAINTS = [
    "My train is delayed by 2 hours. Please provide the running status.",
    "My mobile phone was lost from the train. Please help.",
    "The toilet in my coach is very dirty and there is no water.",
    "The food served in the train is poor quality.",
    "I need medical assistance for a passenger who is feeling sick.",
    "My Tatkal ticket booking is not working on IRCTC.",
]

# ============================================================
# NLP
# ============================================================
@st.cache_resource
def get_nltk_resources():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for resource in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_text(text):
    stop_words, lemmatizer = get_nltk_resources()

    text = str(text)
    text = html.unescape(text)

    # NOTE: the original dict here had duplicate keys (multiple entries that
    # all *displayed* as "â" but were actually different mojibake byte
    # sequences rendered identically by some editors/terminals). Python
    # dict literals silently keep only the LAST value for a repeated key,
    # so several of the intended fixes below never actually ran before.
    # Rewritten with unicode escapes so every mapping is distinct and fires.
    encoding_fixes = {
        "\u00e2\u20ac\u2122": "'",    # â€™  -> '  (right single quote)
        "\u00e2\u20ac\u02dc": "'",    # â€˜  -> '  (left single quote)
        "\u00e2\u20ac\u0153": '"',    # â€œ  -> "  (left double quote)
        "\u00e2\u20ac\u009d": '"',    # â€\x9d -> "  (right double quote)
        "\u00e2\u20ac\u00a6": "...",  # â€¦  -> ... (ellipsis)
        "\u00e2\u20ac\u201c": "-",    # â€“  -> -  (en dash)
        "\u00e2\u20ac\u201d": "-",    # â€”  -> -  (em dash)
    }


    for wrong, correct in encoding_fixes.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\b\d{10}\b", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    negation_words = {"no", "not", "never", "neither", "nor"}

    words = []
    for word in text.split():
        if word in negation_words:
            words.append(word)
        elif word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)


# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource(show_spinner="Training NLP and ML models...")
def train_models(data_file):
    df = pd.read_csv(data_file, encoding="latin-1")

    required_columns = {"Item ID", "Sentiment", "SentimentText"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Clean_Text"] = df["SentimentText"].apply(preprocess_text)

    # A small number of rows (pure @mentions, URLs or numbers) become an
    # empty string after cleaning. TF-IDF technically tolerates this (it
    # just yields a zero vector), but an empty "complaint" carries no
    # signal for either classification or clustering, so we drop these
    # rows and report how many were removed for transparency.
    n_before = len(df)
    df = df[df["Clean_Text"].str.len() > 0].reset_index(drop=True)
    n_dropped_empty = n_before - len(df)

    X_text = df["Clean_Text"]
    y = df["Sentiment"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # -------------------------
    # Baseline SVM
    # -------------------------
    baseline_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_base = baseline_tfidf.fit_transform(X_train_text)
    X_test_base = baseline_tfidf.transform(X_test_text)

    baseline_svm = SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=42,
    )
    baseline_svm.fit(X_train_base, y_train)
    baseline_pred = baseline_svm.predict(X_test_base)

    # -------------------------
    # Baseline Decision Tree
    # -------------------------
    baseline_dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )
    baseline_dt.fit(X_train_base, y_train)
    baseline_dt_pred = baseline_dt.predict(X_test_base)

    # -------------------------
    # Tuned Decision Tree
    # -------------------------
    dt_param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
    }

    dt_grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        param_grid=dt_param_grid,
        cv=StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        ),
        scoring="f1_macro",
        n_jobs=-1,
    )

    dt_grid_search.fit(X_train_base, y_train)
    tuned_dt = dt_grid_search.best_estimator_
    tuned_dt_pred = tuned_dt.predict(X_test_base)

    # -------------------------
    # Tuned SVM
    # -------------------------
    svm_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("svm", SVC(class_weight="balanced", random_state=42)),
        ]
    )

    param_grid = {
        "tfidf__max_features": [3000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "svm__C": [0.5, 1, 5, 10],
        "svm__kernel": ["linear", "rbf"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    grid_search.fit(X_train_text, y_train)
    best_model = grid_search.best_estimator_
    tuned_pred = best_model.predict(X_test_text)

    # Per-class precision/recall/F1 (not just the aggregate macro F1), so
    # rare classes aren't hidden behind one averaged number.
    per_class_report = classification_report(
        y_test, tuned_pred, output_dict=True, zero_division=0
    )

    # -------------------------
    # Final K-Means
    # -------------------------
    kmeans_tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
    )

    X_train_kmeans = kmeans_tfidf.fit_transform(X_train_text)

    final_kmeans = KMeans(
        n_clusters=10,
        random_state=42,
        n_init=10,
    )

    cluster_labels = final_kmeans.fit_predict(X_train_kmeans)
    silhouette = silhouette_score(X_train_kmeans, cluster_labels)

    # 2D visualization
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_kmeans_2d = svd.fit_transform(X_train_kmeans)

    feature_names = kmeans_tfidf.get_feature_names_out()
    cluster_terms = {}

    for cluster_id in range(10):
        center = final_kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-10:][::-1]
        cluster_terms[cluster_id] = [
            feature_names[i] for i in top_indices
        ]

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Complaint_Count")
    )

    # Evaluation
    model_metrics = pd.DataFrame(
        [
            {
                "Model": "Baseline SVM",
                "Accuracy": accuracy_score(y_test, baseline_pred),
                "Macro F1": f1_score(
                    y_test, baseline_pred, average="macro"
                ),
                "CV Macro F1": np.nan,
            },
            {
                "Model": "Tuned SVM",
                "Accuracy": accuracy_score(y_test, tuned_pred),
                "Macro F1": f1_score(
                    y_test, tuned_pred, average="macro"
                ),
                "CV Macro F1": grid_search.best_score_,
            },
            {
                "Model": "Baseline Decision Tree",
                "Accuracy": accuracy_score(y_test, baseline_dt_pred),
                "Macro F1": f1_score(
                    y_test, baseline_dt_pred, average="macro"
                ),
                "CV Macro F1": np.nan,
            },
            {
                "Model": "Tuned Decision Tree",
                "Accuracy": accuracy_score(y_test, tuned_dt_pred),
                "Macro F1": f1_score(
                    y_test, tuned_dt_pred, average="macro"
                ),
                "CV Macro F1": dt_grid_search.best_score_,
            },
        ]
    )

    cm = confusion_matrix(
        y_test,
        tuned_pred,
        labels=sorted(y.unique()),
    )

    return {
        "df": df,
        "X_train_text": X_train_text,
        "X_test_text": X_test_text,
        "y_train": y_train,
        "y_test": y_test,
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "tuned_dt_params": dt_grid_search.best_params_,
        "model_metrics": model_metrics,
        "confusion_matrix": cm,
        "labels": sorted(y.unique()),
        "per_class_report": per_class_report,
        "n_dropped_empty": n_dropped_empty,
        "kmeans": final_kmeans,
        "kmeans_tfidf": kmeans_tfidf,
        "cluster_labels": cluster_labels,
        "cluster_terms": cluster_terms,
        "cluster_counts": cluster_counts,
        "X_kmeans_2d": X_kmeans_2d,
        "silhouette": silhouette,
    }


# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================
KNOWLEDGE_BASE = [
    {
        "topic": "Ticket Booking",
        "category": "Ticketing / Reservation",
        "keywords": ["ticket", "booking", "book", "reservation", "irctc"],
        "answer": "For ticket-booking issues, verify the booking details and the relevant reservation/IRCTC status before taking further action.",
    },
    {
        "topic": "PNR Status",
        "category": "Ticketing / Reservation",
        "keywords": ["pnr", "status", "reservation"],
        "answer": "For a PNR-related query, check the current PNR and reservation status using the applicable railway service.",
    },
    {
        "topic": "RAC and Waiting List",
        "category": "Ticketing / Reservation",
        "keywords": ["rac", "waiting", "waitlist", "waiting list"],
        "answer": "For RAC or waiting-list questions, check the current booking status and applicable reservation information.",
    },
    {
        "topic": "Ticket Cancellation",
        "category": "Ticketing / Reservation",
        "keywords": ["cancel", "cancellation", "refund", "refunds"],
        "answer": "For cancellation or refund questions, verify the applicable ticket conditions and current railway/IRCTC refund rules before advising the passenger.",
    },
    {
        "topic": "Train Running Status",
        "category": "Train Delay / Status",
        "keywords": ["delay", "delayed", "late", "running", "train status", "hour"],
        "answer": "For a delayed train, check the latest train-running status and use the current operational information when responding to the passenger.",
    },
    {
        "topic": "Cleanliness",
        "category": "Coach Facilities / Cleanliness",
        "keywords": ["dirty", "clean", "cleanliness", "garbage", "dustbin"],
        "answer": "The complaint relates to cleanliness. Record the coach/location details and route the issue for appropriate housekeeping or service action.",
    },
    {
        "topic": "Water and Toilet",
        "category": "Coach Facilities / Cleanliness",
        "keywords": ["water", "toilet", "washroom"],
        "answer": "The complaint relates to water or toilet facilities. Capture the coach/location details and route the issue for service attention.",
    },
    {
        "topic": "Air Conditioning",
        "category": "Coach Facilities / Cleanliness",
        "keywords": ["ac", "air conditioning", "fan", "cooling", "charging"],
        "answer": "The complaint relates to coach equipment. Capture the coach details and route the issue for technical/service attention.",
    },
    {
        "topic": "Food Complaint",
        "category": "Catering / Vendors",
        "keywords": ["food", "meal", "vendor", "catering", "pantry"],
        "answer": "The complaint relates to catering or food service. Record the relevant train, coach and vendor details for follow-up.",
    },
    {
        "topic": "Lost Property",
        "category": "Security / Lost Property",
        "keywords": ["lost", "mobile", "phone", "luggage", "bag", "stolen", "missing"],
        "answer": "The complaint relates to lost property or a security concern. Record the relevant journey and item details and route it for appropriate follow-up.",
    },
    {
        "topic": "Security Incident",
        "category": "Security / Lost Property",
        "keywords": ["security", "police", "fir", "robbed", "aggressive"],
        "answer": "The complaint relates to a security incident. Capture the incident details and route it through the appropriate security process.",
    },
    {
        "topic": "Medical Emergency",
        "category": "Medical Assistance",
        "keywords": ["doctor", "medical", "medicine", "pain", "sick", "vomit", "emergency"],
        "answer": "The complaint indicates a medical assistance requirement. Capture the location and urgency details and route it for appropriate medical assistance.",
    },
    {
        "topic": "Complaint Follow-up",
        "category": "Administrative Follow-up",
        "keywords": ["matter", "action", "forwarded", "noted", "reply", "follow", "concerned"],
        "answer": "The query appears to require complaint follow-up. Record the complaint reference and check its current handling status.",
    },
]


def retrieve_knowledge(query, top_k=1):
    q = set(preprocess_text(query).split())

    scored = []
    for item in KNOWLEDGE_BASE:
        keywords = set(item["keywords"])
        overlap = len(q.intersection(keywords))
        scored.append((overlap, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] == 0:
        return {
            "topic": "General Complaint Guidance",
            "category": "Other / General Service",
            "answer": "The query does not strongly match the current prototype knowledge base. Please capture the complaint details and route it for staff review.",
            "source": "Knowledge Base Fallback",
            "score": 0,
        }

    item = scored[0][1]
    return {
        "topic": item["topic"],
        "category": item["category"],
        "answer": item["answer"],
        "source": "Knowledge Base",
        "score": scored[0][0],
    }


# ============================================================
# RAG KNOWLEDGE BASE — FARE / REFUND ("PRICES") ASSISTANT
# ============================================================
# IMPORTANT PROTOTYPE NOTE:
# The figures below are illustrative example values written for this
# academic prototype only. They are NOT live IRCTC / Indian Railways
# tariffs and must not be presented to a real passenger as current,
# official pricing. For a production system this content would be
# retrieved from the official IRCTC/Rail Madad fare API instead of a
# static list.
FARE_KNOWLEDGE_BASE = [
    {
        "topic": "Tatkal Charges",
        "keywords": ["tatkal", "urgent", "last", "minute", "emergency", "booking"],
        "answer": (
            "Tatkal booking carries an additional Tatkal charge on top of the base fare, "
            "typically a percentage of the fare (example range used in this prototype: "
            "~10% for second class, ~30% for other classes, subject to class-wise min/max caps). "
            "Exact current charges should always be confirmed on the official IRCTC fare page."
        ),
    },
    {
        "topic": "Cancellation Charges",
        "keywords": ["cancel", "cancellation", "cancelling"],
        "answer": (
            "Cancellation charges in this prototype scale with how close to departure the "
            "ticket is cancelled and with the class of travel (example bands used here: "
            "flat minimum charge if cancelled well in advance, rising to a larger percentage "
            "of the fare within a few hours of departure). Confirmed, RAC and Waitlisted "
            "tickets follow different example rules in this demo."
        ),
    },
    {
        "topic": "Refund Rules",
        "keywords": ["refund", "money", "back", "reimburse"],
        "answer": (
            "Refunds in this prototype are modelled as (fare paid − applicable cancellation "
            "charge − any convenience fee). Refund timelines and exact percentages shown here "
            "are illustrative example values for demo purposes only."
        ),
    },
    {
        "topic": "Waitlist / RAC Refund",
        "keywords": ["waitlist", "wl", "rac", "waiting"],
        "answer": (
            "If a Waitlisted ticket does not get confirmed, this prototype models a full "
            "example refund (minus a small clerkage charge) when cancelled through the normal "
            "process. RAC tickets follow a separate example refund band in this demo."
        ),
    },
    {
        "topic": "Senior Citizen / Concession Pricing",
        "keywords": ["senior", "citizen", "concession", "discount", "elderly"],
        "answer": (
            "Concession pricing (e.g. senior citizen, student) is modelled in this prototype "
            "as a percentage reduction applied to the base fare before Tatkal/other charges. "
            "The exact percentage shown here is an example value for demonstration only."
        ),
    },
    {
        "topic": "Platform Ticket Pricing",
        "keywords": ["platform", "ticket", "entry"],
        "answer": (
            "Platform ticket pricing in this prototype is modelled as a small flat fee, "
            "used here purely to illustrate how a fixed-price query would be answered by "
            "the RAG assistant."
        ),
    },
    {
        "topic": "Reservation / Convenience Fee",
        "keywords": ["reservation", "fee", "convenience", "surcharge", "charge"],
        "answer": (
            "This prototype models a small reservation/convenience fee added at booking, "
            "on top of the base class fare, purely to demonstrate fee-breakdown style answers."
        ),
    },
]


def retrieve_fare_knowledge(query):
    q = set(preprocess_text(query).split())

    scored = []
    for item in FARE_KNOWLEDGE_BASE:
        keywords = set(item["keywords"])
        overlap = len(q.intersection(keywords))
        scored.append((overlap, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] == 0:
        return {
            "topic": "General Fare Guidance",
            "answer": (
                "This demo fare assistant does not have a matching entry for that query. "
                "In a production system this would fall back to the live IRCTC fare-enquiry API "
                "instead of the static prototype knowledge base."
            ),
            "source": "Fare Knowledge Base Fallback",
            "score": 0,
        }

    item = scored[0][1]
    return {
        "topic": item["topic"],
        "answer": item["answer"],
        "source": "Fare Knowledge Base",
        "score": scored[0][0],
    }


# ============================================================
# PRIORITY RULE (application-layer, not an ML classifier)
# ============================================================
# The original prototype only looked at the predicted CATEGORY to decide
# priority, which meant a mild complaint and a genuinely urgent one in the
# same category always got the same priority. This keyword layer is a
# second, independent rule-based signal that can escalate priority based
# on the complaint's own wording, regardless of category. It is still a
# simple rule (substring match on the cleaned text), not a trained model —
# stated explicitly so it is never mistaken for ML-based urgency detection.
URGENCY_KEYWORDS = [
    "emergency", "urgent", "immediately", "unconscious", "bleeding",
    "fire", "smoke", "assault", "threat", "weapon", "critical",
    "heart attack", "chest pain", "accident", "life threatening",
    "not breathing", "severe", "robbed at",
]


def check_urgency_signal(complaint_text):
    cleaned = preprocess_text(complaint_text)
    for kw in URGENCY_KEYWORDS:
        if preprocess_text(kw) in cleaned:
            return kw
    return None


def estimate_confidence(model, clean_text):
    """Rough, uncalibrated confidence from the SVM decision scores.

    NOTE: this is NOT a calibrated probability (that would require
    probability=True on the SVC, which is expensive to also run inside
    GridSearchCV). It softmaxes the one-vs-rest decision_function scores
    to give a relative sense of how confident vs. borderline a prediction
    is, and should be read as "relative confidence", not a true P(class).
    """
    try:
        scores = model.decision_function([clean_text])[0]
        scores = np.atleast_1d(scores)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        return float(np.max(probs))
    except Exception:
        return None


def analyze_complaint(complaint, artifacts):
    if not complaint or not complaint.strip():
        return None

    model = artifacts["best_model"]
    clean_text = preprocess_text(complaint)

    predicted_label = int(model.predict([clean_text])[0])
    category = CATEGORY_MAPPING.get(
        predicted_label,
        "Other / General Service",
    )
    confidence = estimate_confidence(model, clean_text)

    # Base rule: category-based priority.
    base_high = category in ["Security / Lost Property", "Medical Assistance"]

    # Escalation rule: urgency wording in the complaint itself, independent
    # of predicted category.
    urgency_hit = check_urgency_signal(complaint)

    if base_high:
        priority = "HIGH"
        priority_reason = f"Category '{category}' is treated as high priority."
    elif urgency_hit:
        priority = "HIGH"
        priority_reason = f"Escalated on keyword match: '{urgency_hit}'."
    else:
        priority = "STANDARD"
        priority_reason = "No high-priority category or urgency keyword matched."

    rag = retrieve_knowledge(complaint)

    return {
        "complaint": complaint.strip(),
        "category": category,
        "category_id": predicted_label,
        "confidence": confidence,
        "priority": priority,
        "priority_reason": priority_reason,
        "knowledge_topic": rag["topic"],
        "knowledge_category": rag["category"],
        "response": rag["answer"],
        "response_source": rag["source"],
        "retrieval_score": rag["score"],
    }


# ============================================================
# SESSION DATA (with simple file-based persistence)
# ============================================================
# The original prototype only kept complaints in st.session_state, so the
# entire queue vanished on a page refresh or app restart. For a "management
# system" that's a real gap, so complaints are now also mirrored to a local
# JSON file and reloaded on startup. This is still a lightweight, single-file
# store meant for a prototype/demo — a real deployment would use a proper
# database instead.
import json
import os

COMPLAINTS_STORE_FILE = "complaints_store.json"


def _save_complaints_to_disk():
    try:
        with open(COMPLAINTS_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.complaints, f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Could not save complaint queue to disk: {e}")


def _load_complaints_from_disk():
    if os.path.exists(COMPLAINTS_STORE_FILE):
        try:
            with open(COMPLAINTS_STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def initialize_session():
    if "complaints" not in st.session_state:
        st.session_state.complaints = _load_complaints_from_disk()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "fare_chat_history" not in st.session_state:
        st.session_state.fare_chat_history = []


def add_complaint(result, source="Passenger Portal"):
    complaint_id = f"C{len(st.session_state.complaints) + 1:04d}"

    record = {
        "Complaint ID": complaint_id,
        "Created At": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Source": source,
        "Complaint": result["complaint"],
        "Category": result["category"],
        "Confidence": (
            round(result["confidence"], 3)
            if result.get("confidence") is not None
            else None
        ),
        "Priority": result["priority"],
        "Priority Reason": result.get("priority_reason", ""),
        "Knowledge Topic": result["knowledge_topic"],
        "AI Recommendation": result["response"],
        "Status": "Pending",
        "Staff Action": "",
        "Resolution": "",
    }

    st.session_state.complaints.append(record)
    _save_complaints_to_disk()
    return complaint_id


def load_demo_complaints(artifacts):
    if st.session_state.complaints:
        return

    for text in SAMPLE_COMPLAINTS:
        result = analyze_complaint(text, artifacts)
        add_complaint(result, source="Demo Data")


# ============================================================
# APP
# ============================================================
initialize_session()

try:
    artifacts = train_models(DATA_FILE)
except Exception as e:
    st.error(
        "The application could not load/train the models. "
        f"Check that `{DATA_FILE}` is in the same folder as this app."
    )
    st.exception(e)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🚆 RailAssist")
st.sidebar.caption("Railway Complaint Management Prototype")

portal = st.sidebar.radio(
    "Select Portal",
    ["👤 Passenger Portal", "👨‍💼 Staff Portal"],
)

if portal == "👤 Passenger Portal":
    page = st.sidebar.radio(
        "Passenger Menu",
        ["🏠 Home", "📝 Submit Complaint", "💬 AI Assistant", "💰 Fare & Refund Chat"],
    )
else:
    page = st.sidebar.radio(
        "Staff Menu",
        [
            "📊 Dashboard",
            "📋 Complaint Queue",
            "🔵 Cluster Analytics",
            "🤖 Model Performance",
            "🧾 Prompts & Methodology",
        ],
    )

st.sidebar.divider()
st.sidebar.caption("Prototype inspired by railway complaint-management workflows.")
st.sidebar.caption("Status / resolution fields are application-level fields.")

# ============================================================
# PASSENGER HOME
# ============================================================
if portal == "👤 Passenger Portal" and page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🚆 RailAssist</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">A Rail Madad-inspired ML prototype for railway complaint '
        'management, using NLP, classification and knowledge-based retrieval</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This is an academic prototype inspired by railway complaint-management workflows. "
        "It is not an official railway service."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("ML Model", "Tuned SVM")
    with c2:
        st.metric("Knowledge Layer", "Retrieval (RAG-style)")
    with c3:
        st.metric("Unsupervised Analytics", "K-Means")

    st.caption(
        "Knowledge retrieval here uses keyword matching over a static knowledge "
        "base (no embeddings, no generative LLM) — see Staff Portal → Prompts & "
        "Methodology for the full design note."
    )

    st.markdown("### What can you do?")

    a, b = st.columns(2)

    with a:
        st.markdown("#### 📝 Submit a Complaint")
        st.write(
            "Describe your railway complaint. The system predicts a complaint "
            "category and provides a knowledge-grounded recommendation."
        )

    with b:
        st.markdown("#### 💬 Ask the AI Assistant")
        st.write(
            "Ask questions about complaint-related topics such as train delays, "
            "ticketing, cleanliness, lost property and medical assistance."
        )

    st.markdown("### System Flow")

    st.code(
        """
Passenger Complaint
       ↓
NLP Preprocessing
       ↓
TF-IDF + Tuned SVM
       ↓
Category + Application Priority Rule
       ↓
Knowledge Retrieval
       ↓
Grounded Response
       ↓
Passenger / Staff Interface
        """,
        language="text",
    )

# ============================================================
# PASSENGER SUBMIT COMPLAINT
# ============================================================
elif portal == "👤 Passenger Portal" and page == "📝 Submit Complaint":

    st.markdown(
        '<div class="main-title">📝 Submit a Complaint</div>',
        unsafe_allow_html=True,
    )

    complaint = st.text_area(
        "Describe your complaint",
        height=160,
        placeholder=(
            "Example: My train is delayed by 2 hours. "
            "Please provide the latest running status."
        ),
    )

    if st.button("🚀 Analyze & Submit Complaint", use_container_width=True):

        if not complaint.strip():
            st.warning("Please enter a complaint before submitting.")
        else:
            result = analyze_complaint(complaint, artifacts)
            complaint_id = add_complaint(result)

            st.success(f"Complaint submitted successfully. Reference ID: {complaint_id}")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Predicted Category", result["category"])

            with c2:
                conf = result.get("confidence")
                st.metric(
                    "Model Confidence",
                    f"{conf:.0%}" if conf is not None else "N/A",
                )

            with c3:
                st.metric("Priority", result["priority"])

            with c4:
                st.metric("Knowledge Topic", result["knowledge_topic"])

            st.caption(
                "Confidence is an uncalibrated relative score derived from the "
                "SVM's decision function (softmax of one-vs-rest scores), not a "
                "true calibrated probability."
            )

            st.markdown("### 🤖 AI Recommendation")
            st.write(result["response"])

            st.caption(
                f"Response source: {result['response_source']} | "
                f"Knowledge match score: {result['retrieval_score']}"
            )

            st.markdown("### 🔎 Processing Summary")
            st.write(
                f"**ML Category:** {result['category']}  \n"
                f"**Priority:** {result['priority']} — {result['priority_reason']}  \n"
                f"**Knowledge Topic:** {result['knowledge_topic']}  \n"
                f"**Knowledge Category:** {result['knowledge_category']}"
            )

# ============================================================
# PASSENGER AI ASSISTANT
# ============================================================
elif portal == "👤 Passenger Portal" and page == "💬 AI Assistant":

    st.markdown(
        '<div class="main-title">💬 Railway AI Assistant</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Ask a question and the system retrieves a relevant topic from the prototype knowledge base."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input(
        "Ask about train delays, ticketing, cleanliness, lost property..."
    )

    if user_question:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        result = retrieve_knowledge(user_question)

        assistant_text = (
            f"**Topic:** {result['topic']}\n\n"
            f"{result['answer']}\n\n"
            f"*Source: {result['source']}*"
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_text}
        )

        st.rerun()

# ============================================================
# PASSENGER FARE & REFUND CHAT (RAG, "prices" requirement)
# ============================================================
elif portal == "👤 Passenger Portal" and page == "💰 Fare & Refund Chat":

    st.markdown(
        '<div class="main-title">💰 Fare & Refund Assistant</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Ask about Tatkal charges, cancellation, refunds, concessions and other "
        "price-related topics. Answers are retrieved from a prototype fare "
        "knowledge base using the same keyword-based retrieval as the AI Assistant "
        "(RAG-style architecture, not embedding-based retrieval or LLM generation)."
    )

    st.warning(
        "Demo data only. Figures are illustrative examples for this prototype, "
        "not live IRCTC/Indian Railways tariffs."
    )

    for message in st.session_state.fare_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    fare_question = st.chat_input(
        "Ask about Tatkal charges, cancellation, refunds, concessions..."
    )

    if fare_question:
        st.session_state.fare_chat_history.append(
            {"role": "user", "content": fare_question}
        )

        result = retrieve_fare_knowledge(fare_question)

        assistant_text = (
            f"**Topic:** {result['topic']}\n\n"
            f"{result['answer']}\n\n"
            f"*Source: {result['source']} | Match score: {result['score']}*"
        )

        st.session_state.fare_chat_history.append(
            {"role": "assistant", "content": assistant_text}
        )

        st.rerun()

# ============================================================
# STAFF DASHBOARD
# ============================================================
elif portal == "👨‍💼 Staff Portal" and page == "📊 Dashboard":

    st.markdown(
        '<div class="main-title">📊 Staff Dashboard</div>',
        unsafe_allow_html=True,
    )

    if st.button("➕ Load Demo Complaints"):
        load_demo_complaints(artifacts)
        st.success("Demo complaints loaded.")

    complaints_df = pd.DataFrame(st.session_state.complaints)

    if complaints_df.empty:
        st.info(
            "No complaints are currently in the application queue. "
            "Submit a complaint from the Passenger Portal or load demo complaints."
        )
    else:
        total = len(complaints_df)
        high = int((complaints_df["Priority"] == "HIGH").sum())
        pending = int((complaints_df["Status"] == "Pending").sum())
        resolved = int((complaints_df["Status"] == "Resolved").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Complaints", total)
        c2.metric("High Priority", high)
        c3.metric("Pending", pending)
        c4.metric("Resolved", resolved)

        st.markdown("### Complaint Category Distribution")

        category_counts = (
            complaints_df["Category"]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Count")
        )

        st.bar_chart(
            category_counts.set_index("Category")
        )

        st.markdown("### Recent Complaint Queue")
        st.dataframe(
            complaints_df[
                [
                    "Complaint ID",
                    "Created At",
                    "Complaint",
                    "Category",
                    "Priority",
                    "Status",
                ]
            ].tail(10),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# STAFF COMPLAINT QUEUE
# ============================================================
elif portal == "👨‍💼 Staff Portal" and page == "📋 Complaint Queue":

    st.markdown(
        '<div class="main-title">📋 Complaint Queue</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.complaints:
        st.info("No complaints available. Submit or load a complaint first.")
    else:
        complaints_df = pd.DataFrame(st.session_state.complaints)

        search = st.text_input(
            "🔎 Search complaints",
            placeholder="Search by complaint, category or ID...",
        )

        priority_filter = st.selectbox(
            "Priority",
            ["All", "HIGH", "STANDARD"],
        )

        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "In Progress", "Resolved", "Escalated"],
        )

        filtered = complaints_df.copy()

        if search:
            mask = (
                filtered.astype(str)
                .apply(
                    lambda col: col.str.contains(
                        search,
                        case=False,
                        na=False,
                    )
                )
                .any(axis=1)
            )
            filtered = filtered[mask]

        if priority_filter != "All":
            filtered = filtered[
                filtered["Priority"] == priority_filter
            ]

        if status_filter != "All":
            filtered = filtered[
                filtered["Status"] == status_filter
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Complaint Details / Staff Action")

        if not filtered.empty:

            selected_id = st.selectbox(
                "Select Complaint",
                filtered["Complaint ID"].tolist(),
            )

            index = next(
                i
                for i, item in enumerate(st.session_state.complaints)
                if item["Complaint ID"] == selected_id
            )

            selected = st.session_state.complaints[index]

            st.write("**Complaint:**", selected["Complaint"])
            st.write("**Predicted Category:**", selected["Category"])
            if selected.get("Confidence") is not None:
                st.write("**Model Confidence:**", f"{selected['Confidence']:.0%}")
            st.write("**Priority:**", selected["Priority"])
            if selected.get("Priority Reason"):
                st.caption(f"Priority reason: {selected['Priority Reason']}")
            st.write("**Knowledge Topic:**", selected["Knowledge Topic"])

            st.info(selected["AI Recommendation"])

            new_status = st.selectbox(
                "Status",
                ["Pending", "In Progress", "Resolved", "Escalated"],
                index=[
                    "Pending",
                    "In Progress",
                    "Resolved",
                    "Escalated",
                ].index(selected["Status"]),
            )

            staff_action = st.text_area(
                "Staff Action",
                value=selected["Staff Action"],
                placeholder="Describe the action taken by staff...",
            )

            resolution = st.text_area(
                "Resolution",
                value=selected["Resolution"],
                placeholder="Enter the resolution details...",
            )

            if st.button("💾 Update Complaint"):
                st.session_state.complaints[index]["Status"] = new_status
                st.session_state.complaints[index]["Staff Action"] = staff_action
                st.session_state.complaints[index]["Resolution"] = resolution
                _save_complaints_to_disk()
                st.success("Complaint updated successfully.")
                st.rerun()

# ============================================================
# K-MEANS ANALYTICS
# ============================================================
elif portal == "👨‍💼 Staff Portal" and page == "🔵 Cluster Analytics":

    st.markdown(
        '<div class="main-title">🔵 K-Means Complaint Cluster Analysis</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "K-Means is used here for exploratory staff analytics. "
        "It discovers recurring text themes without using the predefined labels."
    )

    c1, c2 = st.columns(2)
    c1.metric("Clusters", "10")
    c2.metric("Silhouette Score", f"{artifacts['silhouette']:.4f}")

    st.warning(
        "The low silhouette score indicates substantial overlap between clusters. "
        "Therefore, cluster themes should be interpreted as exploratory rather than definitive."
    )

    left, right = st.columns([1.5, 1])

    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            artifacts["X_kmeans_2d"][:, 0],
            artifacts["X_kmeans_2d"][:, 1],
            c=artifacts["cluster_labels"],
            alpha=0.7,
        )
        ax.set_xlabel("SVD Component 1")
        ax.set_ylabel("SVD Component 2")
        ax.set_title("K-Means Complaint Clusters")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### Cluster Distribution")
        st.dataframe(
            artifacts["cluster_counts"],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Cluster Themes")

    cluster_rows = []

    for cluster_id in range(10):
        cluster_rows.append(
            {
                "Cluster": cluster_id,
                "Complaint Count": int(
                    artifacts["cluster_counts"]
                    .loc[
                        artifacts["cluster_counts"]["Cluster"] == cluster_id,
                        "Complaint_Count",
                    ]
                    .iloc[0]
                ),
                "Top Terms": ", ".join(
                    artifacts["cluster_terms"][cluster_id]
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(cluster_rows),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif portal == "👨‍💼 Staff Portal" and page == "🤖 Model Performance":

    st.markdown(
        '<div class="main-title">🤖 ML Model Performance</div>',
        unsafe_allow_html=True,
    )

    metrics_df = artifacts["model_metrics"].copy()

    for column in ["Accuracy", "Macro F1", "CV Macro F1"]:
        metrics_df[column] = metrics_df[column].round(4)

    if artifacts.get("n_dropped_empty", 0) > 0:
        st.caption(
            f"Note: {artifacts['n_dropped_empty']} row(s) in train.csv became an "
            "empty string after text cleaning (pure @mentions/URLs/numbers) and "
            "were excluded from training and evaluation."
        )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Per-Class Performance (Tuned SVM)")

    st.caption(
        "The aggregate Macro F1 above hides how the model does on individual, "
        "especially rare, categories. This table breaks it down class by class "
        "on the held-out test set."
    )

    report = artifacts["per_class_report"]
    per_class_rows = []
    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        category_name = CATEGORY_MAPPING.get(int(label), f"Class {label}")
        per_class_rows.append(
            {
                "Category ID": label,
                "Category": category_name,
                "Precision": round(stats["precision"], 3),
                "Recall": round(stats["recall"], 3),
                "F1": round(stats["f1-score"], 3),
                "Support (test rows)": int(stats["support"]),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows).sort_values("Support (test rows)")
    st.dataframe(per_class_df, use_container_width=True, hide_index=True)

    st.markdown("### Tuned SVM Parameters")

    st.json(artifacts["best_params"])

    st.markdown("### Tuned Decision Tree Parameters")

    st.json(artifacts["tuned_dt_params"])

    st.markdown("### Tuned SVM Confusion Matrix")

    cm = artifacts["confusion_matrix"]
    labels = artifacts["labels"]

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Tuned SVM Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Current System Architecture")

    st.code(
        """
Complaint
   ↓
NLP Preprocessing
   ↓
TF-IDF
   ↓
Tuned SVM ──────────────→ Category
   │
   └→ Priority Rule
   ↓
Knowledge Retrieval
   ↓
Grounded Recommendation
   ↓
Passenger / Staff Portal
        """,
        language="text",
    )

    st.caption(
        "Priority is currently application-rule based, not a separately trained ML priority classifier."
    )

# ============================================================
# PROMPTS & METHODOLOGY
# ============================================================
elif portal == "👨‍💼 Staff Portal" and page == "🧾 Prompts & Methodology":

    st.markdown(
        '<div class="main-title">🧾 Prompts & Methodology</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Prompts used to build the fare/refund knowledge base")

    st.warning(
        "Honesty note: this content was **authored/drafted with an LLM**, not "
        "collected or scraped from a real data source. If asked where the fare "
        "data 'came from', the accurate answer is that it was generated from the "
        "prompt below and hand-edited — it was not extracted from any dataset."
    )

    st.write(
        "The ML classifier (SVM/Decision Tree) is trained directly on `train.csv` and "
        "does not use LLM prompts. The **Fare & Refund knowledge base** (used by the "
        "'💰 Fare & Refund Chat' assistant) was drafted with the help of an LLM "
        "using the prompt template below, then reviewed and hand-edited before being "
        "hard-coded into `FARE_KNOWLEDGE_BASE`."
    )

    st.code(
        """SYSTEM / TASK PROMPT:
You are helping build a demo knowledge base for a student railway-complaint
prototype called RailAssist. Generate 6-8 short knowledge-base entries about
Indian Railways ticket pricing topics (Tatkal charges, cancellation charges,
refund rules, waitlist/RAC refunds, concession pricing, platform tickets,
reservation fees).

For each entry return:
- topic: short title
- keywords: 3-6 lowercase words a passenger might type
- answer: 2-3 sentences explaining the general RULE or STRUCTURE of the
  charge (not exact rupee figures, since this is a demo and figures must
  not be presented as official/current pricing)

Clearly mark that these are illustrative example values for an academic
prototype, not live IRCTC tariffs, and that a production system should call
the official IRCTC/Rail Madad fare API instead of a static list.

Return the result as a JSON list of objects with keys: topic, keywords, answer.""",
        language="text",
    )

    st.caption(
        "This is documented here for transparency/assessment purposes, per the "
        "assignment requirement to state the prompts used to extract/generate data."
    )

    st.markdown("### 2. Model performance across varied hyperparameters")

    st.write(
        "`GridSearchCV` with 5-fold stratified cross-validation was used to search "
        "the hyperparameter grid below for the SVM pipeline (scoring = macro F1):"
    )

    st.code(
        """param_grid = {
    "tfidf__max_features": [3000, 5000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "svm__C": [0.5, 1, 5, 10],
    "svm__kernel": ["linear", "rbf"],
}""",
        language="python",
    )

    st.write(
        "An equivalent grid search was run for the Decision Tree "
        "(`criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`). "
        "The best combination found for each model, and how baseline vs. tuned "
        "models compare, are shown below (same figures as the Model Performance page):"
    )

    metrics_df = artifacts["model_metrics"].copy()
    for column in ["Accuracy", "Macro F1", "CV Macro F1"]:
        metrics_df[column] = metrics_df[column].round(4)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Best SVM hyperparameters**")
        st.json(artifacts["best_params"])
    with c2:
        st.markdown("**Best Decision Tree hyperparameters**")
        st.json(artifacts["tuned_dt_params"])

    st.markdown("### 3. Knowledge retrieval design note")

    st.write(
        "Both chat assistants (complaint knowledge base and fare/refund knowledge base) "
        "use lightweight keyword-overlap retrieval rather than embedding-based vector "
        "search or LLM generation — this is a **RAG-inspired retrieval architecture**, "
        "not full RAG (there is no generation step; answers are picked, not written). "
        "It stays fully offline and free to run for this prototype. A production "
        "version could swap the retrieval step for a sentence-embedding similarity "
        "search and add an LLM generation step on top of the retrieved context."
    )

    st.markdown("### 4. Priority rule and confidence score")

    st.write(
        "Priority combines two application-level rules, not a trained classifier: "
        "(1) the predicted category (Security/Medical → HIGH), and (2) an "
        "independent keyword check on the complaint's own wording (e.g. "
        "'emergency', 'bleeding', 'fire') that can escalate any category to HIGH. "
        "The 'Model Confidence' shown on the Submit Complaint page is an "
        "uncalibrated relative score from the SVM's decision function (softmax "
        "of one-vs-rest scores) — useful for spotting borderline predictions, "
        "but not a true calibrated probability."
    )

    st.markdown("### 5. Data persistence")

    st.write(
        "Submitted complaints are saved to a local `complaints_store.json` file "
        "in addition to `st.session_state`, so the queue survives an app restart "
        "or browser refresh. This is a simple file-based store for prototype "
        "purposes, not a production database."
    )
