# app.py — AI Customer Support Assistant & Insights Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import random
import re
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support Assistant",
    layout="wide",
    page_icon="💬"
)

# ----------------------------------------------------
# CUSTOM CSS — PROFESSIONAL UI
# ----------------------------------------------------
st.markdown("""
<style>
html, body, .main {
    background-color: #0f172a !important;
    color: #02080f !important;
    font-family: "Segoe UI", sans-serif;
}
h1, h2, h3 {
    color: #02080f !important;
    font-weight: 600;
}
hr {
    border: 1px solid #1f2937;
}
.kpi-card {
    background: #020617;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    box-shadow: 0px 6px 20px rgba(15, 23, 42, 0.7);
}
.kpi-label {
    font-size: 13px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #e5e7eb;
}
.kpi-delta {
    font-size: 13px;
    margin-top: 4px;
}
.positive { color: #22c55e; }
.negative { color: #f97316; }
.neutral  { color: #9ca3af; }
.small-note {
    font-size: 12px;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)


def html_kpi(label, value, delta=None, delta_color="neutral"):
    color_map = {
        "positive": "#22c55e",
        "negative": "#f97316",
        "neutral": "#9ca3af"
    }
    color = color_map.get(delta_color, "#9ca3af")

    html = f"""
    <div style="
        background: #0c162c;
        padding: 16px 22px;
        border-radius: 14px;
        border: 1px solid #1e293b;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.35);
        width: 100%;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 13px; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px;">
            {label}
        </div>
        <div style="font-size: 32px; font-weight: 700; color: #f1f5f9; margin-top: 4px;">
            {value}
        </div>
        {f'<div style="font-size: 12px; color: {color}; margin-top: 4px;">{delta}</div>' if delta else ""}
    </div>
    """
    st.components.v1.html(html, height=130)

# ----------------------------------------------------
# LOAD DATA & MODELS
# ----------------------------------------------------
@st.cache_data
def load_data():
    # Use your local CSV path
    df = pd.read_csv(r"bitext_customer_support.csv")
    df = df.dropna(subset=["instruction", "intent"])
    df = df.drop_duplicates(subset=["instruction"])

    df["clean_text"] = (
        df["instruction"]
        .astype(str)
        .str.replace(r"[^A-Za-z0-9\s]", " ", regex=True)
        .str.lower()
        .str.strip()
    )

    # Synthetic timestamps if not present
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    df["timestamp"] = start_date + pd.to_timedelta(
        np.random.randint(0, 90, len(df)), unit="D"
    )

    return df


@st.cache_resource
@st.cache_resource
def load_models():
    clf = joblib.load(r"intent_classifier.pkl")
    vec = joblib.load(r"tfidf_vectorizer.pkl")

    # Small talk model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- FIXED PART ----
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu"  # runs on CPU
    )

    generator = pipeline(
        "text-generation",
        model=qwen_model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=0.0,      # deterministic
        do_sample=False,      # no random sampling
        pad_token_id=tokenizer.eos_token_id
    )

    sentiment = pipeline("sentiment-analysis")

    return clf, vec, embedder, generator, sentiment

data = load_data()
model, tfidf, embedder, generator, sentiment_analyzer = load_models()

# ----------------------------------------------------
# SMALL TALK
# ----------------------------------------------------
smalltalk_bank = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "thanks": ["thanks", "thank you", "appreciate it"],
    "goodbye": ["bye", "goodbye", "see you", "take care"]
}
smalltalk_vecs = {k: embedder.encode(v) for k, v in smalltalk_bank.items()}


def detect_smalltalk(text, threshold=0.55):
    vec = embedder.encode([text])
    best, score = None, 0
    for key, mat in smalltalk_vecs.items():
        s = np.max(cosine_similarity(vec, mat))
        if s > score:
            best, score = key, s
    return best if score >= threshold else None


def smalltalk_reply(tag):
    if tag == "greeting":
        # Guided greeting with main issue types
        return (
            "Hello! 👋 How can I help you today?\n\n"
            "Here are some common things I can assist with:\n"
            "- 👤 Account & login\n"
            "- 🧾 Orders (place, cancel, change)\n"
            "- 📦 Delivery & tracking\n"
            "- 💳 Payments & refunds\n"
            "- ✉️ Any other questions about your purchase"
        )
    if tag == "thanks":
        return random.choice([
            "You're very welcome! Happy to help 🤗",
            "Anytime! I'm here whenever you need.",
            "My pleasure!"
        ])
    if tag == "goodbye":
        return random.choice([
            "Goodbye! Take care 👋",
            "See you soon!",
            "Bye-bye! Have a wonderful day!"
        ])
    return None

# ----------------------------------------------------
# INTENT PREDICTION
# ----------------------------------------------------
def predict_intent(user_query: str):
    q_vec = tfidf.transform([user_query])
    probs = model.predict_proba(q_vec)[0]
    idx = np.argmax(probs)
    intent = model.classes_[idx]
    confidence = float(probs[idx])
    return intent, confidence

# ----------------------------------------------------
# TONE & CONTEXT HELPERS
# ----------------------------------------------------
def detect_tone_from_text(user_text: str) -> str:
    """
    Simple heuristic tone detection:
    - Greeting-like → Friendly
    - Very short / command-like → Short & direct
    - Otherwise → Professional
    """
    text = user_text.strip().lower()
    if re.match(r"^(hi|hey|hello)\b", text):
        return "friendly"
    if len(text.split()) <= 5:
        return "short and direct"
    return "professional"


def build_conversation_context(max_turns: int = 3) -> str:
    """
    Build a short context string from the last few chat turns.
    """
    hist = st.session_state.get("chat_history", [])
    lines = []
    for role, msg in hist[-max_turns * 2:]:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg}")
    return "\n".join(lines)

# ----------------------------------------------------
# STRUCTURE / CLEAN HELPERS
# ----------------------------------------------------
def build_paraphrase_prompt(base, tone, context):
    return f"""
You are a customer support chatbot for a small online shop.

Rewrite the message below into a clean, concise reply (3–6 short lines).  
Rules:
- No placeholders like {{order}}, {{email}}, {{phone}}. Replace with "your order" or "your account".
- No dramatic words ("decoded", "restitution", "sensing", etc.).
- No made-up content or assumptions.
- No marketing language.
- NO long paragraphs.
- NO repeating lines.
- Keep tone: {tone}.
- If base message is unclear, rewrite it into the simplest helpful version.
- Use plain human language.
- If context clarifies user intent, adjust wording accordingly.

Conversation context:
{context}

Message to rewrite:
\"\"\"{base}\"\"\"

Final rewritten message:
"""


def clean_generated_reply(text, prompt=None):
    # Remove prompt echo
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Remove unwanted boilerplate
    text = re.sub(r"[\[\(].*?[\]\)]", "", text)  # remove brackets like [insert...]
    text = re.sub(r"(affiliate|reddit|post was submitted|api key|irc)", "", text, flags=re.I)
    text = re.sub(r"\byour order\s+your order\b", "your order", flags=re.I)
    text = re.sub(r"[^A-Za-z0-9.,?!\n ]", " ", text)

    # Normalize sentence breaks
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Remove repeated lines
    lines = []
    seen = set()
    for line in text.split("\n"):
        l = line.strip()
        if l and l.lower() not in seen:
            lines.append(l)
            seen.add(l.lower())

    return "\n".join(lines).strip()
    

def safe_reply(text: str) -> str:
    banned = ["amazon", "reddit", "affiliate", "post was submitted"]
    for b in banned:
        if b in text.lower():
            return (
                "I'm sorry, that wording wasn't very clear. "
                "Here is a simpler version: I'm here to help you with your order, account, "
                "delivery, or refund. Could you share a few more details?"
            )
    return text


def enforce_formatting(reply: str, intent: str) -> str:
    # Extract sentences
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]\s+", reply)
        if len(s.strip()) > 2
    ][:5]

    if not sentences:
        return reply

    # Refund → bullets
    if "refund" in intent:
        return "\n".join(f"- {s}" for s in sentences)

    # Password, tracking, order changes → steps
    if any(x in intent for x in ["recover", "track", "change", "cancel"]):
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    # Otherwise → clean short reply
    return "\n".join(sentences)


def clean_template_reply(text: str) -> str:
    """
    Clean raw template responses from the dataset:
    - remove placeholders
    - fix 'your order your order' style double phrases
    - keep it simple and neutral
    """
    # Replace the most common placeholders explicitly
    text = re.sub(r"\{\{Order Number\}\}", "your order number", text)
    text = re.sub(r"\{\{.*?\}\}", "your order", text)  # fallback for other placeholders

    # Fix ugly repetitions like "your order your order"
    text = re.sub(r"\byour order\s+your order\b", "your order", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bpurchase with the number your order\b",
        "purchase with your order number",
        text,
        flags=re.IGNORECASE
    )

    # Strip overly dramatic or weird wording
    bad_phrases = [
        "I've decoded that",
        "I'm sensing that",
        "honored to assist",
        "quest for restitution",
        "restitution"
    ]
    for p in bad_phrases:
        text = text.replace(p, "")

    # Normalize spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

# ----------------------------------------------------
# HYBRID REPLY (INTENT + TEMPLATE + PHI-3 MINI)
# ----------------------------------------------------
def generate_hybrid_reply(user_query: str):
    """
    Main improved hybrid chatbot:
    - small talk
    - intent prediction
    - category override
    - context-aware template selection
    - strict Qwen rewriting
    - formatting (bullets/steps)
    - fallback cleaning
    - agent handoff logic
    """

    # ---------- 0. Small talk ----------
    st_tag = detect_smalltalk(user_query)
    if st_tag:
        reply = smalltalk_reply(st_tag)
        return reply, "smalltalk", None

    # ---------- 1. Predict intent ----------
    intent, confidence = predict_intent(user_query)

    # ---------- 1.1 Category lock ----------
    category = st.session_state.get("issue_category")
    if category:
        if category == "payments":
            intent = "payment_issue"
        elif category == "delivery":
            intent = "track_order"
        elif category == "orders":
            intent = "place_order"
        elif category == "account":
            intent = "recover_password"
        elif category == "other":
            intent = intent  # keep original

    # ---------- 1.2 Agent handoff ----------
    agent_pending = st.session_state.get("agent_pending_issue", False)
    if intent == "contact_human_agent":
        if not agent_pending:
            # We first need the user to describe the issue
            st.session_state["agent_pending_issue"] = True
            reply = (
                "I can connect you to a human agent.\n"
                "Before I do, please tell me briefly what you need help with "
                "(order, delivery, payment, account, or something else)."
            )
            return reply, intent, confidence
        else:
            # User already described issue → handoff completed
            st.session_state["agent_pending_issue"] = False
            reply = (
                "Thank you for clarifying.\n"
                "I'm now handing this over to a human agent who can assist you further."
            )
            return reply, intent, confidence

    # ---------- 2. Select template from dataset ----------
    subset = data[data["intent"] == intent]
    if not subset.empty:
        base = subset["response"].sample(1, random_state=42).values[0]
    else:
        base = (
            "I'm here to help with your order, delivery, payment, or account. "
            "Could you share a bit more detail?"
        )

    # ---------- 3. Clean template ----------
    base = clean_template_reply(base)

    # ---------- 4. Context-based overrides ----------
    # Look at recent conversation to "fix" wrong template meaning
    last_messages = [msg for role, msg in st.session_state.get("chat_history", []) if role == "user"]
    last_user = last_messages[-1].lower() if last_messages else ""

    if "damaged" in user_query.lower() and "refund" in user_query.lower():
        base = "Let me help you with a refund for a damaged item."
    if "change" in user_query.lower() and "payment" in user_query.lower():
        base = "You want to update your payment method. Here’s how."
    if "track" in user_query.lower() and "order" in user_query.lower():
        base = "Here’s how you can see the current status and tracking updates for your order."
    if "password" in user_query.lower() or "login" in user_query.lower():
        base = "Let me help you reset access to your account."

    # ---------- 5. Build rewriting prompt ----------
    context = build_conversation_context(max_turns=3)
    tone = detect_tone_from_text(user_query)
    prompt = build_paraphrase_prompt(base, tone, context)

    # ---------- 6. LLM REWRITE ----------
    try:
        gen = generator(prompt)
        raw = gen[0]["generated_text"]

        # Remove prompt echo
        if raw.startswith(prompt):
            raw = raw[len(prompt):].strip()

        rewritten = clean_generated_reply(raw, prompt)

    except Exception:
        rewritten = base  # fallback

    # ---------- 7. Fallback if blank/too short ----------
    if not rewritten or len(rewritten.split()) < 4:
        rewritten = base

    # ---------- 8. Apply formatting based on intent ----------
    final = enforce_formatting(rewritten, intent)

    # ---------- 9. Final safety cleaning ----------
    final = safe_reply(final)

    return final.strip(), intent, confidence

# ----------------------------------------------------
# ANALYTICS HELPERS
# ----------------------------------------------------
def plot_top_intents(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    top_intents = df["intent"].value_counts().head(10)
    sns.barplot(x=top_intents.index, y=top_intents.values, ax=ax, color="#38bdf8")
    ax.set_title("Top 10 Customer Support Intents")
    ax.set_xlabel("Intent")
    ax.set_ylabel("Number of Tickets")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig


def plot_category_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    cat_counts = df["category"].value_counts()
    ax.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%")
    ax.set_title("Share of Queries by Category")
    return fig


def plot_time_trend(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    weekly = df.set_index("timestamp").resample("W")["instruction"].count()
    weekly.index = weekly.index.strftime("%Y-%m-%d")
    sns.lineplot(x=weekly.index, y=weekly.values, marker="o", ax=ax, color="#a855f7")
    ax.set_title("Weekly Volume of Customer Tickets")
    ax.set_xlabel("Week")
    ax.set_ylabel("Tickets")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig


def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    crosstab = pd.crosstab(df["category"], df["intent"])
    sns.heatmap(crosstab, cmap="viridis", ax=ax)
    ax.set_title("Intent–Category Relationship")
    ax.set_xlabel("Intent")
    ax.set_ylabel("Category")
    plt.xticks(rotation=90, fontsize=6)
    fig.tight_layout()
    return fig


def plot_wordcloud(df, intent):
    subset = df[df["intent"] == intent]
    text = " ".join(subset["clean_text"].astype(str).values)
    if not text:
        text = "no data"
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Common Terms for Intent: {intent}")
    fig.tight_layout()
    return fig

# ----------------------------------------------------
# BUSINESS INSIGHTS HELPERS
# ----------------------------------------------------
def compute_business_insights(df):
    now = df["timestamp"].max()
    last_week = now - timedelta(days=7)
    prev_week = last_week - timedelta(days=7)

    this_week = df[df["timestamp"] >= last_week]
    prev_week_df = df[(df["timestamp"] < last_week) & (df["timestamp"] >= prev_week)]

    # Top issues this week
    top_issues = this_week["intent"].value_counts().head(3)

    # Pattern-based masks (not hard-coded mapping)
    refund_mask = df["intent"].str.contains("refund", case=False, regex=True)
    delivery_mask = df["intent"].str.contains("delivery|track_order|shipping", case=False, regex=True)

    this_refund = this_week[refund_mask].shape[0]
    prev_refund = prev_week_df[refund_mask].shape[0]

    this_delivery = this_week[delivery_mask].shape[0]
    prev_delivery = prev_week_df[delivery_mask].shape[0]

    def pct_change(curr, prev):
        if prev == 0:
            return 100.0 if curr > 0 else 0.0
        return ((curr - prev) / prev) * 100.0

    refund_delta = pct_change(this_refund, prev_refund)
    delivery_delta = pct_change(this_delivery, prev_delivery)

    # Sentiment on sample
    if len(this_week) > 0:
        sample = this_week["instruction"].sample(min(80, len(this_week)), random_state=42)
        sentiments = sentiment_analyzer(sample.to_list())
    else:
        sentiments = []

    pos = sum(1 for s in sentiments if s["label"].lower().startswith("pos"))
    neg = sum(1 for s in sentiments if s["label"].lower().startswith("neg"))
    neu = len(sentiments) - pos - neg
    total = len(sentiments) if len(sentiments) > 0 else 1

    sent_scores = {
        "positive": round(100 * pos / total, 1),
        "negative": round(100 * neg / total, 1),
        "neutral": round(100 * neu / total, 1),
    }

    # Narrative summary
    summary_parts = []
    if len(top_issues) > 0:
        main_intent = top_issues.index[0]
        summary_parts.append(f"Most customer tickets this week are about **{main_intent}**.")
    if refund_delta > 15:
        summary_parts.append("Refund-related tickets increased compared to last week.")
    elif refund_delta < -10:
        summary_parts.append("Refund-related tickets decreased, which is a good sign.")
    if delivery_delta > 15:
        summary_parts.append("Delivery or tracking issues went up this week.")
    elif delivery_delta < -10:
        summary_parts.append("Delivery and tracking issues went down compared to last week.")
    if sent_scores["negative"] > 25:
        summary_parts.append(
            "Negative sentiment is relatively high. It may be worth reviewing recent changes or promotions."
        )

    if not summary_parts:
        summary = "This week looks stable overall. No major spikes or drops across key issue types."
    else:
        summary = " ".join(summary_parts)

    # Recommendations (simple rules)
    recs = []
    if refund_delta > 15:
        recs.append(
            "Take a sample of refund-related tickets and look for repeated reasons "
            "(e.g., product quality, sizing, late delivery)."
        )
        recs.append(
            "Consider updating your refund policy page or adding clearer messaging on the checkout page."
        )
    if delivery_delta > 15:
        recs.append(
            "Check performance of your shipping partners and tracking links. "
            "Make sure tracking emails are sent on time."
        )
    if sent_scores["negative"] > 25:
        recs.append(
            "Review negative messages manually and tag common themes such as price complaints, "
            "quality issues, or delays."
        )
    if len(top_issues) > 0 and "recover_password" in top_issues.index[0]:
        recs.append(
            "Password recovery is a top issue. Simplify the reset flow or add help text around login."
        )
    if not recs:
        recs.append("Keep monitoring ticket volumes and sentiment. Current patterns look stable.")

    # Weekly report subset for export
    weekly_export = this_week.copy()

    return {
        "top_issues": top_issues,
        "refund_current": this_refund,
        "refund_prev": prev_refund,
        "refund_delta": refund_delta,
        "delivery_current": this_delivery,
        "delivery_prev": prev_delivery,
        "delivery_delta": delivery_delta,
        "sentiment": sent_scores,
        "summary": summary,
        "recommendations": recs,
        "weekly_export": weekly_export
    }

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.title("💬 Customer Support AI")
st.sidebar.markdown(
    """
This app shows a prototype of:

- Intent classification (ML)
- Hybrid response generation (templates + Qwen2.5-1.5B-Instruct)
- Small talk detection
- Analytics & business insights for small e-commerce
"""
)
st.sidebar.markdown("---")
st.sidebar.write("Tickets in dataset:", len(data))
st.sidebar.write("Unique intents:", data["intent"].nunique())

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab_chat, tab_analytics, tab_insights = st.tabs(
    ["💬 Chat Assistant", "📊 Support Analytics", "📈 Business Insights"]
)

# =========================
# TAB 1 – CHAT ASSISTANT
# =========================
with tab_chat:
    st.title("💬 AI Customer Support Assistant")

    # Initialize memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_user_message" not in st.session_state:
        st.session_state.last_user_message = None
    if "issue_category" not in st.session_state:
        st.session_state.issue_category = None
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = True

    # END CHAT BUTTON
    end_col, start_col = st.columns([1, 1])

    with end_col:
        if st.button("🚪 End Chat"):
            st.session_state.chat_active = False
            st.session_state.chat_history.append(
                (
                    "assistant",
                    "Thanks for chatting with us! 👋 If you need anything else, "
                    "feel free to start a new chat anytime."
                )
            )

    with start_col:
        if not st.session_state.chat_active:
            if st.button("🔄 Start New Chat"):
                st.session_state.chat_history = []
                st.session_state.last_user_message = None
                st.session_state.issue_category = None
                st.session_state.chat_active = True
                st.rerun()

    # If chat ended → show messages but disable input
    if not st.session_state.chat_active:
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)
        st.stop()

    # --- Issue category quick buttons (always visible) ---
    st.markdown("#### Common topics")

    bc1, bc2, bc3, bc4, bc5 = st.columns(5)

    def set_category(cat, message):
        st.session_state.issue_category = cat
        st.session_state.chat_history.append(("assistant", message))

    with bc1:
        if st.button("👤 Account"):
            set_category("account", "Sure — let's work on your account or login issue. What exactly is happening?")
    with bc2:
        if st.button("🧾 Orders"):
            set_category("orders", "Got it — this is about an order. How can I help with your order?")
    with bc3:
        if st.button("📦 Delivery"):
            set_category("delivery", "Okay — sounds like a delivery or tracking issue. What's happening with your delivery?")
    with bc4:
        if st.button("💳 Payments"):
            set_category("payments", "Understood — we’re looking at a payment or refund issue. Tell me what you’d like to update or fix.")
    with bc5:
        if st.button("❓ Other"):
            set_category("other", "No problem — tell me briefly what you'd like help with.")

    # --- Show chat history ---
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # --- Regenerate last answer ---
    can_regen = any(role == "assistant" for role, _ in st.session_state.chat_history)
    regen_clicked = False
    if can_regen:
        regen_clicked = st.button("🔁 Regenerate last response")

    # --- User input box ---
    user_input = st.chat_input("Describe your issue, or ask a question...")

    # Handle regenerate
    if regen_clicked and st.session_state.last_user_message:
        with st.chat_message("assistant"):
            with st.spinner("Regenerating..."):
                reply, intent, confidence = generate_hybrid_reply(
                    st.session_state.last_user_message
                )
                st.session_state.chat_history.append(("assistant", reply))
                st.markdown(reply)

                if intent != "smalltalk" and confidence is not None:
                    st.markdown("---")
                    st.markdown(f"**Predicted intent:** `{intent}`")
                    st.markdown(f"**Confidence:** `{confidence*100:.1f}%`")
                    st.progress(min(1.0, confidence + 0.05))
        st.rerun()

    # Handle new user input
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.last_user_message = user_input

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, intent, confidence = generate_hybrid_reply(user_input)
                st.session_state.chat_history.append(("assistant", reply))
                st.markdown(reply)

                if intent != "smalltalk" and confidence is not None:
                    st.markdown("---")
                    st.markdown(f"**Predicted intent:** `{intent}`")
                    st.markdown(f"**Confidence:** `{confidence*100:.1f}%`")
                    st.progress(min(1.0, confidence + 0.05))

        st.rerun()

# =========================
# TAB 2 – ANALYTICS
# =========================
with tab_analytics:
    st.title("📊 Customer Support Analytics")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Intents")
        fig1 = plot_top_intents(data)
        st.pyplot(fig1)

    with col2:
        st.subheader("Category Distribution")
        fig2 = plot_category_distribution(data)
        st.pyplot(fig2)

    st.subheader("Weekly Ticket Volume")
    fig3 = plot_time_trend(data)
    st.pyplot(fig3)

    st.subheader("Intent–Category Heatmap")
    fig4 = plot_heatmap(data)
    st.pyplot(fig4)

    st.subheader("Word Cloud by Intent")
    intents_sorted = sorted(data["intent"].unique())
    selected_intent = st.selectbox("Choose an intent", intents_sorted)
    fig5 = plot_wordcloud(data, selected_intent)
    st.pyplot(fig5)

# =========================
# TAB 3 – BUSINESS INSIGHTS
# =========================
with tab_insights:
    st.title("📈 Business Insights Dashboard")

    insights = compute_business_insights(data)
    top_issues = insights["top_issues"]

    # Summary narrative
    st.markdown("### 📝 What changed this week?")
    st.markdown(insights["summary"])
    st.markdown("<hr/>", unsafe_allow_html=True)

    # 🔥 TOP ISSUES (KPI CARDS)
    st.markdown("### 🔥 Most Frequent Customer Problems (This Week)")
    k1, k2, k3 = st.columns(3)
    top_list = list(top_issues.items())

    if len(top_list) > 0:
        with k1:
            html_kpi(label=top_list[0][0], value=top_list[0][1])
    if len(top_list) > 1:
        with k2:
            html_kpi(label=top_list[1][0], value=top_list[1][1])
    if len(top_list) > 2:
        with k3:
            html_kpi(label=top_list[2][0], value=top_list[2][1])

    # 📉 REFUND + DELIVERY MONITORS
    st.markdown("---")
    st.markdown("### 📉 Refund & 🚚 Delivery Issue Monitors")

    c1, c2 = st.columns(2)

    with c1:
        delta_txt = f"Change vs last week: {insights['refund_delta']:+.1f}%"
        delta_color = "negative" if insights["refund_delta"] > 0 else "positive"
        html_kpi(
            label="Refund-related tickets (this week)",
            value=insights["refund_current"],
            delta=delta_txt,
            delta_color=delta_color
        )

    with c2:
        delta_txt = f"Change vs last week: {insights['delivery_delta']:+.1f}%"
        delta_color = "negative" if insights["delivery_delta"] > 0 else "positive"
        html_kpi(
            label="Delivery / tracking tickets (this week)",
            value=insights["delivery_current"],
            delta=delta_txt,
            delta_color=delta_color
        )

    # 💬 SENTIMENT CARDS
    st.markdown("---")
    st.markdown("### 💬 Sentiment of Recent Customer Messages")

    s_pos = insights["sentiment"]["positive"]
    s_neg = insights["sentiment"]["negative"]
    s_neu = insights["sentiment"]["neutral"]

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        html_kpi("Positive", f"{s_pos}%")
    with sc2:
        html_kpi("Neutral", f"{s_neu}%")
    with sc3:
        html_kpi("Negative", f"{s_neg}%")

    # 📤 EXPORT CSV
    st.markdown("---")
    st.markdown("### 📤 Export Weekly Report")

    weekly_df = insights["weekly_export"].copy()
    weekly_csv = weekly_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Export weekly report as CSV",
        data=weekly_csv,
        file_name="weekly_support_report.csv",
        mime="text/csv"
    )

    st.markdown(
        '<p class="small-note">Includes all customer messages from the last 7 days.</p>',
        unsafe_allow_html=True
    )

    # 🎯 RECOMMENDATIONS
    st.markdown("---")
    st.markdown("### 🎯 Recommendations for the Business Owner")

    for rec in insights["recommendations"]:
        st.markdown(f"- {rec}")

