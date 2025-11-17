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
    font-family: "Segoe UI", sans-serif;#02080f
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
    df = pd.read_csv(r"bitext_customer_support.csv")
    df = df.dropna(subset=["instruction", "intent"])
    df = df.drop_duplicates(subset=["instruction"])

    df["clean_text"] = (
        df["instruction"]
        .astype(str)
        .str.replace(r"[^A-Za-z0-9\\s]", "", regex=True)
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
def load_models():
    clf = joblib.load(r"intent_classifier.pkl")
    vec = joblib.load(r"tfidf_vectorizer.pkl")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        pad_token_id=50256,
        eos_token_id=50256
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
        return random.choice([
            "Hey there! 👋 How can I help you today?",
            "Hello! Glad you're here 😊 What would you like to do?",
            "Hi! How may I assist you?"
        ])
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
# PREDICT INTENT
# ----------------------------------------------------
def predict_intent(user_query: str):
    q_vec = tfidf.transform([user_query])
    probs = model.predict_proba(q_vec)[0]
    idx = np.argmax(probs)
    intent = model.classes_[idx]
    confidence = float(probs[idx])
    return intent, confidence

# ----------------------------------------------------
# STRUCTURED REPLY FORMATTER
# ----------------------------------------------------
def enforce_structure(raw_text: str) -> str:
    """
    Try to keep numbered lists / bullets and remove junk/repeats.
    """
    # Strip weird headers / model prompt bits
    raw_text = raw_text.split("Base reply:", 1)[0]
    raw_text = raw_text.replace("Intent:", "").replace("intent:", "")

    # Remove URLs and junk phrases
    raw_text = re.sub(r"http[s]?://\\S+", "", raw_text)
    raw_text = re.sub(r"(Read more|Affiliate|Post was submitted)", "", raw_text, flags=re.IGNORECASE)

    # Normalize spaces
    raw_text = re.sub(r"\\s{2,}", " ", raw_text).strip()

    # If it accidentally collapsed everything, just return as-is
    if len(raw_text) < 10:
        return raw_text

    # Make sure line breaks between list items
    raw_text = raw_text.replace("1.", "\\n1.").replace("2.", "\\n2.") \
                       .replace("3.", "\\n3.").replace("4.", "\\n4.") \
                       .replace("5.", "\\n5.")

    # Remove obvious repeated trailing segments
    sentences = raw_text.split(". ")
    seen = set()
    cleaned_sentences = []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            cleaned_sentences.append(s.strip())
            seen.add(key)
    structured = ". ".join(cleaned_sentences).strip()

    return structured


def safe_reply(text: str) -> str:
    banned = ["amazon", "reddit", "affiliate", "post was submitted"]
    for b in banned:
        if b in text.lower():
            return (
                "I'm sorry, that wording wasn't very clear. "
                "Here is a simpler version: I'm here to help you with your order, account, or refund. "
                "Could you share a few more details?"
            )
    return text

# ----------------------------------------------------
# HYBRID REPLY (MODEL + TEMPLATE + GENERATION)
# ----------------------------------------------------
def generate_hybrid_reply(user_query: str):
    # 1. Small talk gate
    st_tag = detect_smalltalk(user_query)
    if st_tag:
        return smalltalk_reply(st_tag), "smalltalk", None

    # 2. Predict intent & confidence
    intent, confidence = predict_intent(user_query)

    # 3. Template from dataset
    subset = data[data["intent"] == intent]
    if not subset.empty:
        base = subset["response"].sample(1).values[0]
    else:
        base = "I'm here to help, could you please tell me a bit more about your issue?"

    base = re.sub(r"\\{\\{.*?\\}\\}", "", base)

    # 4. Stronger, structured prompt
    prompt = f"""
Intent: {intent}
User query: {user_query}

You are a polite, concise customer support agent.
Rewrite the reply below in a friendly, CLEAR, structured format.

Rules:
- Use short sentences.
- If steps are needed, use a numbered list (1., 2., 3.).
- If explaining options, use bullet points (-).
- Max 5 bullet/numbered items.
- No external links, no blog-style text.
- No 'post', 'submitted', 'affiliate', 'Reddit'.
- Stay under 8 lines.

Base reply: {base}
"""

    # 5. Generate
    out = generator(
        prompt,
        max_new_tokens=120,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        pad_token_id=50256,
        eos_token_id=50256
    )
    raw = out[0]["generated_text"]

    # Extract only after "Base reply" section if model echoes prompt
    raw = raw.split("Base reply:", 1)[-1] if "Base reply:" in raw else raw

    reply = enforce_structure(raw)
    reply = safe_reply(reply)

    # 6. Fallbacks for blank / too-short answers
    if reply is None or len(reply.strip()) < 10:
        reply = base.strip()
        reply = enforce_structure(reply)

    return reply, intent, confidence

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
        summary_parts.append("Negative sentiment is relatively high. It may be worth reviewing recent changes or promotions.")

    if not summary_parts:
        summary = "This week looks stable overall. No major spikes or drops across key issue types."
    else:
        summary = " ".join(summary_parts)

    # Recommendations (simple rules)
    recs = []
    if refund_delta > 15:
        recs.append("Take a sample of refund-related tickets and look for repeated reasons (e.g., product quality, sizing, late delivery).")
        recs.append("Consider updating your refund policy page or adding clearer messaging on the checkout page.")
    if delivery_delta > 15:
        recs.append("Check performance of your shipping partners and tracking links. Make sure tracking emails are sent on time.")
    if sent_scores["negative"] > 25:
        recs.append("Review negative messages manually and tag common themes such as price complaints, quality issues, or delays.")
    if len(top_issues) > 0 and "recover_password" in top_issues.index[0]:
        recs.append("Password recovery is a top issue. Simplify the reset flow or add help text around login.")
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
- Hybrid response generation (templates + GPT-2)
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

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show chat history
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    user_input = st.chat_input("Ask a question about your order, account, refund, or delivery...")
    if user_input:
        # Show user message
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, intent, confidence = generate_hybrid_reply(user_input)

                st.session_state.chat_history.append(("assistant", reply))
                st.markdown(reply)

                if intent != "smalltalk":
                    st.markdown("---")
                    st.markdown(f"**Predicted intent:** `{intent}`")
                    if confidence is not None:
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

    # ---------------------------
    # 🔥 TOP ISSUES (KPI CARDS)
    # ---------------------------
    st.markdown("### 🔥 Most Frequent Customer Problems (This Week)")

    k1, k2, k3 = st.columns(3)

    top_list = list(top_issues.items())

    if len(top_list) > 0:
        with k1:
            html_kpi(
                label=top_list[0][0],
                value=top_list[0][1]
            )
    if len(top_list) > 1:
        with k2:
            html_kpi(
                label=top_list[1][0],
                value=top_list[1][1]
            )
    if len(top_list) > 2:
        with k3:
            html_kpi(
                label=top_list[2][0],
                value=top_list[2][1]
            )

    # ---------------------------
    # 📉 REFUND + DELIVERY MONITORS
    # ---------------------------
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

    # ---------------------------
    # 💬 SENTIMENT CARDS
    # ---------------------------
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

    # ---------------------------
    # 📤 EXPORT CSV
    # ---------------------------
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

    # ---------------------------
    # 🎯 RECOMMENDATIONS
    # ---------------------------
    st.markdown("---")
    st.markdown("### 🎯 Recommendations for the Business Owner")

    for rec in insights["recommendations"]:
        st.markdown(f"- {rec}")

