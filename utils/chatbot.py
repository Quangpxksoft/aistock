# # chatbot.py


import json
import os
from difflib import get_close_matches

TABS = ["forecast", "risk", "backtest_perf", "rebalance", "optimize", "report"]

FAQ_DATA = {}
ALL_FAQS = []
ALL_QUESTIONS = []
Q_MAP = {}
KEYWORD_MAP = {}

def load_all_faqs():
    global ALL_FAQS, ALL_QUESTIONS, Q_MAP, KEYWORD_MAP

    if ALL_FAQS:
        return ALL_FAQS

    all_faqs = []

    for tab in TABS:
        if tab in FAQ_DATA:
            data = FAQ_DATA[tab]
        else:
            path = os.path.join("knowledge_base", f"{tab}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                FAQ_DATA[tab] = data

        if "sections" not in data:
            continue

        for section in data["sections"].values():
            all_faqs.extend(section.get("faqs", []))

    all_questions = []
    q_map = {}
    keyword_map = {}
    for faq in all_faqs:
        for q in faq.get("questions", []):
            all_questions.append(q)
            q_map[q] = faq.get("answer", "")
            keyword_map[q] = [kw.lower() for kw in faq.get("keywords", [])]

    ALL_FAQS = all_faqs
    ALL_QUESTIONS = all_questions
    Q_MAP = q_map
    KEYWORD_MAP = keyword_map

    return ALL_FAQS

def highlight_keywords(answer: str, keywords: list, user_input: str) -> str:
    highlighted = answer
    for kw in keywords:
        if kw.lower() in user_input.lower():
            highlighted = highlighted.replace(kw, f"**{kw}**")
    return highlighted

def chatbot_answer(user_input: str) -> str:
    faqs = load_all_faqs()
    if not faqs:
        return "❌ Không có dữ liệu FAQ nào để trả lời."

    user_input_lower = user_input.lower()

    # 1️⃣ Exact match (so với câu hỏi)
    for q in ALL_QUESTIONS:
        if user_input_lower == q.lower():
            return Q_MAP[q]

    # 2️⃣ Keyword match
    for faq in faqs:
        keywords = faq.get("keywords", [])
        if any(kw.lower() in user_input_lower for kw in keywords):
            answer = highlight_keywords(faq.get("answer", "❓ Không tìm thấy câu trả lời."), keywords, user_input)
            return answer

    # 3️⃣ Fuzzy match (so gần đúng với câu hỏi)
    match = get_close_matches(user_input, ALL_QUESTIONS, n=1, cutoff=0.5)
    if match:
        answer = Q_MAP[match[0]]
        # Gợi ý thêm 3 câu liên quan
        suggestions = get_close_matches(user_input, ALL_QUESTIONS, n=4, cutoff=0.4)
        suggestions = [s for s in suggestions if s != match[0]]
        if suggestions:
            answer += "\n\n👉 Bạn có thể quan tâm đến:\n" + "\n".join(f"- {s}" for s in suggestions)
        return answer

    return "❓ Xin lỗi, tôi không tìm thấy thông tin phù hợp trong hệ thống."
