# RailAssist – Railway Complaint Management Prototype

## Files

- `app.py` – Streamlit application
- `train.csv` – required dataset
- `requirements.txt` – Python dependencies

## Run

Open a terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application provides:

### Passenger Portal
- Home
- Complaint submission
- ML category prediction
- Application-level priority routing
- Knowledge-grounded recommendation
- Conversational AI-style knowledge assistant
- Fare & Refund Chat: a second RAG-style chat assistant for price-related
  queries (Tatkal charges, cancellation, refunds, concessions, etc.)

### Staff Portal
- Dashboard
- Complaint queue
- Search/filter
- Status/action/resolution fields
- K-Means cluster analytics
- Cluster distribution
- Model performance
- Confusion matrix
- Tuned SVM parameters
- Prompts & Methodology: documents the LLM prompt used to draft the fare
  knowledge base, the hyperparameter grids searched for the SVM and
  Decision Tree models, and a short RAG design note

## Important project notes

1. The application is an academic prototype inspired by railway complaint-management workflows. It is not an official railway system.
2. The original dataset contains `Item ID`, `Sentiment`, and `SentimentText`. In this project, the numeric `Sentiment` field is treated as the classification label; it should not automatically be described as a validated sentiment-analysis target.
3. Priority is currently an application-layer rule based on predicted category. It is not a separately trained priority classifier.
4. Status, Staff Action and Resolution are application-level fields; they are not original dataset columns.
5. The knowledge base in this prototype is a small demonstration knowledge base. For authoritative real-world policy claims, replace it with verified official source documents.
6. K-Means is exploratory. The current K=10 silhouette score is low, so cluster themes should not be treated as definitive categories.
7. The Fare & Refund knowledge base (used by the "Fare & Refund Chat" page) contains illustrative example figures written for this prototype. They are not live IRCTC/Indian Railways tariffs and must not be presented as current, official pricing. A production system should call the official IRCTC/Rail Madad fare API instead.
8. "RAG" in this project refers to a RAG-inspired keyword retrieval architecture (no embeddings, no generation step, no LLM at runtime) — not full retrieval-augmented generation. See Staff Portal -> Prompts & Methodology for the full design note.
9. Complaints now persist across restarts via a local complaints_store.json file (created after the first complaint is submitted). This is a simple file-based store for prototype purposes, not a production database.

## Fixes applied after review

- Fixed a duplicate-key bug in the mojibake/encoding-fix dictionary in preprocess_text (several intended replacements were silently never firing).
- Rows that become an empty string after text cleaning are now dropped before training/evaluation, and the count is shown on the Model Performance page.
- Added a per-class precision/recall/F1 table (Model Performance page) so rare-category performance isn't hidden behind the aggregate Macro F1.
- Added an uncalibrated relative confidence score (from the SVM decision function) shown alongside each prediction.
- Priority is now decided by two independent rules: the predicted category, and a keyword-based urgency check on the complaint text itself (previously category alone decided everything).
- Complaints are now persisted to complaints_store.json, so the queue survives an app restart or browser refresh (previously session-only).
- Reworded "RAG" and "sentiment analysis" references app-wide to accurately describe what the prototype does (keyword-based retrieval; category classification on a repurposed sentiment-labeled dataset), instead of implying embedding-based RAG or validated sentiment analysis.
- Clarified that the Fare & Refund knowledge base was authored/drafted with an LLM prompt, not collected/extracted from a real dataset.
