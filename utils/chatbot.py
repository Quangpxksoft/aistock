
import json
import os
from difflib import get_close_matches

TABS = ["forecast", "risk", "backtest_perf", "rebalance", "optimize", "report", "general"]

FAQ_DATA      = {}
ALL_FAQS      = []
ALL_QUESTIONS = []
Q_MAP         = {}
KEYWORD_MAP   = {}


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
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            FAQ_DATA[tab] = data

        if "sections" not in data:
            continue
        for section in data["sections"].values():
            all_faqs.extend(section.get("faqs", []))

    all_questions = []
    q_map         = {}
    keyword_map   = {}

    for faq in all_faqs:
        for q in faq.get("questions", []):
            all_questions.append(q)
            q_map[q]       = faq.get("answer", "")
            keyword_map[q] = [kw.lower() for kw in faq.get("keywords", [])]

    ALL_FAQS      = all_faqs
    ALL_QUESTIONS = all_questions
    Q_MAP         = q_map
    KEYWORD_MAP   = keyword_map

    return ALL_FAQS


def _score_keyword_match(faq: dict, user_input_lower: str) -> float:
    """
    Tính điểm keyword match có trọng số thay vì match any-keyword.
    Trả về tỷ lệ keyword khớp / tổng keyword — [0.0, 1.0].
    Tránh FAQ có nhiều keyword chung match nhầm.
    """
    keywords = faq.get("keywords", [])
    if not keywords:
        return 0.0
    matched = sum(1 for kw in keywords if kw.lower() in user_input_lower)
    return matched / len(keywords)


def _get_representative_question(faq: dict) -> str:
    """Lấy câu hỏi đại diện (câu đầu tiên) của một FAQ."""
    questions = faq.get("questions", [])
    return questions[0] if questions else ""


def _build_suggestion_response(header: str, faqs: list, max_suggestions: int = 5) -> dict:
    """
    Tạo response dạng gợi ý để người dùng click chọn thay vì gõ cả câu.

    Returns
    -------
    dict với keys:
        "type"        : "suggestion"
        "message"     : chuỗi header hiển thị
        "suggestions" : list[dict] gồm {"label": câu hỏi đại diện, "answer": câu trả lời}
    """
    suggestions = []
    seen_answers = set()
    for faq in faqs:
        answer = faq.get("answer", "")
        # Loại trùng answer (nhiều FAQ cùng trỏ về 1 nội dung)
        if answer in seen_answers:
            continue
        seen_answers.add(answer)
        label = _get_representative_question(faq)
        if label:
            suggestions.append({"label": label, "answer": answer})
        if len(suggestions) >= max_suggestions:
            break

    return {
        "type":        "suggestion",
        "message":     header,
        "suggestions": suggestions,
    }


def chatbot_answer(user_input: str, chat_history: list | None = None) -> str | dict:
    """
    Trả lời câu hỏi dựa trên FAQ.

    Thứ tự ưu tiên:
    1.   Exact match câu hỏi
    1.5. Token match — khớp chính xác từng token với keyword (dành cho
         thuật ngữ ngắn như rsi, lstm, p/e, msi...)
         - 1 kết quả  → trả lời trực tiếp
         - Nhiều kết quả → trả về dict gợi ý để người dùng click chọn
    2.   Keyword match có trọng số (≥ 40% keyword khớp)
    3.   Fuzzy match string similarity (cutoff=0.5)
    4.   Fallback gợi ý câu hỏi gần nhất (cutoff=0.3)
    5.   Không tìm thấy gì

    Parameters
    ----------
    user_input   : câu hỏi / từ khoá người dùng nhập
    chat_history : list các dict {"role", "content"} — dùng để bổ sung
                   context khi câu hỏi hiện tại quá ngắn (< 5 ký tự)

    Returns
    -------
    str  : câu trả lời trực tiếp
    dict : {"type": "suggestion", "message": str, "suggestions": list[dict]}
           khi có nhiều gợi ý để người dùng click chọn
    """
    faqs = load_all_faqs()
    if not faqs:
        return "❌ Không có dữ liệu FAQ nào để trả lời."

    user_input_stripped = user_input.strip()
    user_input_lower    = user_input_stripped.lower()

    # Nếu câu hỏi quá ngắn, bổ sung context từ lượt trước
    if len(user_input_lower) < 5 and chat_history:
        for msg in reversed(chat_history):
            if msg.get("role") == "Bạn" and msg.get("content") != user_input:
                user_input_lower = msg["content"].lower() + " " + user_input_lower
                break

    # ── 1️⃣  Exact match ────────────────────────────────────────────────────
    for q in ALL_QUESTIONS:
        if user_input_lower == q.lower():
            return Q_MAP[q]

    # ── 1.5️⃣  Token match (thuật ngữ ngắn) ────────────────────────────────
    # Tách input thành token, so sánh chính xác (==) với từng keyword của FAQ.
    # Không dùng substring/contains để tránh match nhầm (vd: "ma" match "macd").
    tokens = [t for t in user_input_lower.split() if t]  # loại token rỗng

    if tokens:
        token_matched_faqs  = []   # FAQ có ít nhất 1 token khớp chính xác keyword
        token_matched_score = {}   # faq index → số token khớp (ưu tiên khớp nhiều)

        for idx, faq in enumerate(faqs):
            kw_set = {kw.lower() for kw in faq.get("keywords", [])}
            matched_tokens = sum(1 for t in tokens if t in kw_set)
            if matched_tokens > 0:
                token_matched_faqs.append(faq)
                token_matched_score[idx] = matched_tokens

        if token_matched_faqs:
            # Sắp xếp theo số token khớp giảm dần
            token_matched_faqs.sort(
                key=lambda faq: token_matched_score[faqs.index(faq)],
                reverse=True,
            )

            if len(token_matched_faqs) == 1:
                # Chỉ 1 kết quả → trả lời thẳng
                return token_matched_faqs[0].get("answer", "❓ Không tìm thấy câu trả lời.")

            # Nhiều kết quả → gợi ý để người dùng click chọn
            return _build_suggestion_response(
                header=f"🔍 Tìm thấy {len(token_matched_faqs)} kết quả liên quan đến «{user_input_stripped}». Bạn muốn hỏi về:",
                faqs=token_matched_faqs,
            )

    # ── 2️⃣  Keyword match có trọng số ──────────────────────────────────────
    best_faq   = None
    best_score = 0.0
    for faq in faqs:
        score = _score_keyword_match(faq, user_input_lower)
        if score > best_score:
            best_score = score
            best_faq   = faq

    if best_faq is not None and best_score >= 0.4:
        return best_faq.get("answer", "❓ Không tìm thấy câu trả lời.")

    # ── 3️⃣  Fuzzy match ─────────────────────────────────────────────────────
    match = get_close_matches(user_input_stripped, ALL_QUESTIONS, n=1, cutoff=0.5)
    if match:
        answer      = Q_MAP[match[0]]
        suggestions = get_close_matches(user_input_stripped, ALL_QUESTIONS, n=4, cutoff=0.4)
        suggestions = [s for s in suggestions if s != match[0]][:3]
        if suggestions:
            answer += "\n\n👉 Bạn có thể quan tâm:\n" + "\n".join(f"- {s}" for s in suggestions)
        return answer

    # ── 4️⃣  Fallback — gợi ý dù không đủ cutoff ────────────────────────────
    suggestions = get_close_matches(user_input_stripped, ALL_QUESTIONS, n=3, cutoff=0.3)
    if suggestions:
        return (
            "❓ Tôi chưa tìm thấy thông tin phù hợp.\n\n"
            "💡 Bạn có thể thử hỏi:\n" + "\n".join(f"- {s}" for s in suggestions)
        )

    # ── 5️⃣  Không tìm thấy gì ───────────────────────────────────────────────
    return "❓ Xin lỗi, tôi không tìm thấy thông tin phù hợp trong hệ thống."