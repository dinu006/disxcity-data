import os
import json
import re
import difflib
import requests
import PyPDF2

OLLAMA_URL = "http://localhost:11434/api/generate"

# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def call_ollama(prompt, model_name, temperature=0.2, num_ctx=8192, num_predict=4096):
    """Send a prompt to the local Ollama server and return the raw text response."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return ""


# ---------------------------------------------------------------------------
# JSON extraction (robust to markdown fences / truncation)
# ---------------------------------------------------------------------------

def extract_json(raw_text):
    """Parse a JSON object/array from a model response, tolerating markdown
    fences and minor formatting noise. Returns None on failure."""
    clean_text = raw_text.strip()

    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\n?", "", clean_text)
        clean_text = re.sub(r"\n?```$", "", clean_text)

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def extract_list_field(data, keys):
    """Given parsed JSON, find the first list stored under any of `keys`."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


# ---------------------------------------------------------------------------
# Live logging helper
# ---------------------------------------------------------------------------

def print_questions(questions, label=""):
    """Pretty-print each generated question to the console as soon as it exists."""
    if not questions:
        print(f"  [{label}] No questions returned in this call.")
        return

    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            print(f"  [{label} #{i}] <invalid item: {q!r}>")
            continue
        qtext = q.get("question", "<no question text>")
        options = q.get("options", [])
        correct = q.get("correctAnswer", None)
        print(f"  [{label} #{i}] {qtext}")
        for oi, opt in enumerate(options):
            marker = "\u2714" if oi == correct else " "
            print(f"      {marker} {oi}: {opt}")
        if q.get("explanation"):
            print(f"      explanation: {q['explanation']}")
        print()


# ---------------------------------------------------------------------------
# Step 1: treat each page as its own chunk (no merging across pages)
# ---------------------------------------------------------------------------

def build_page_chunks(pages_text, min_chars=80):
    """Return one chunk per page, skipping pages with almost no text
    (e.g. covers). Each page is sent to the LLM independently."""
    chunks = []

    for i, text in enumerate(pages_text):
        stripped = text.strip()
        if len(stripped) < min_chars:
            print(f"Page {i + 1} has insufficient text ({len(stripped)} chars). Skipping.")
            continue

        chunks.append({"text": stripped, "pages": [i + 1]})

    return chunks


# ---------------------------------------------------------------------------
# Step 2: extract every important fact/concept from a chunk
# ---------------------------------------------------------------------------

def extract_important_facts(chunk_text, model_name):
    prompt = f"""
    Read the following text extracted from a school textbook.
    List EVERY important, testable fact, definition, date, name, cause/effect
    relationship, or concept in it. Be exhaustive - do not limit yourself to a
    fixed number. Each item should be a short, self-contained statement (one
    fact per item), specific enough that a question could be written about it.

    Return ONLY a valid JSON object of this exact structure:
    {{
      "facts": ["Fact 1 statement", "Fact 2 statement", "..."]
    }}

    Text:
    {chunk_text}
    """
    raw = call_ollama(prompt, model_name, temperature=0.1, num_predict=2048)
    data = extract_json(raw)
    facts = extract_list_field(data, ["facts"])
    # Keep only non-empty string facts
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


# ---------------------------------------------------------------------------
# Step 3a: generate one question per extracted fact (guarantees coverage)
# ---------------------------------------------------------------------------

def generate_questions_from_facts(facts, chunk_text, model_name, batch_size=8):
    """Ask the model to write one MCQ per fact, batching facts to keep
    prompts manageable. Returns a flat list of question dicts."""
    all_questions = []

    for start in range(0, len(facts), batch_size):
        batch = facts[start:start + batch_size]
        facts_block = "\n".join(f"{idx + 1}. {fact}" for idx, fact in enumerate(batch))

        prompt = f"""
        You are writing a multiple-choice quiz based on a school textbook passage.
        Below is the original passage for context, followed by a numbered list of
        facts. Write exactly ONE multiple-choice question per fact, testing that
        specific fact. Keep the facts' original order.

        Passage:
        {chunk_text}

        Facts:
        {facts_block}

        Return ONLY a valid JSON object of this exact structure:
        {{
          "questions": [
            {{
              "question": "Question text here?",
              "options": ["Option A", "Option B", "Option C", "Option D"],
              "correctAnswer": 0,
              "explanation": "Brief explanation of the answer."
            }}
          ]
        }}
        The "questions" array must have exactly {len(batch)} items, in the same
        order as the facts listed above.
        """
        raw = call_ollama(prompt, model_name, temperature=0.2, num_predict=4096)
        data = extract_json(raw)
        questions = extract_list_field(data, ["questions", "quiz", "data"])

        batch_num = (start // batch_size) + 1
        print(f"    -- Fact batch {batch_num} results --")
        print_questions(questions, label=f"fact-batch-{batch_num}")

        all_questions.extend(questions)

    return all_questions


# ---------------------------------------------------------------------------
# Step 3b: general free-form pass, catches anything the fact list missed
# ---------------------------------------------------------------------------

def generate_general_questions(chunk_text, model_name):
    prompt = f"""
    Read the following text extracted from a school textbook and generate as
    many high-quality multiple-choice questions as the content supports.
    Cover different facts than a simple summary would - include details,
    numbers, causes/effects, and definitions.

    Return ONLY a valid JSON object matching this exact structure:
    {{
      "questions": [
        {{
          "question": "Question text here?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "correctAnswer": 0,
          "explanation": "Brief explanation of the answer."
        }}
      ]
    }}

    Text:
    {chunk_text}
    """
    raw = call_ollama(prompt, model_name, temperature=0.3, num_predict=4096)
    data = extract_json(raw)
    questions = extract_list_field(data, ["questions", "quiz", "data"])

    print("    -- General pass results --")
    print_questions(questions, label="general")

    return questions


# ---------------------------------------------------------------------------
# Deduplication across the whole chapter
# ---------------------------------------------------------------------------

def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def is_near_duplicate(question_text, seen_normalized, threshold=0.85):
    norm = normalize(question_text)
    for existing in seen_normalized:
        if difflib.SequenceMatcher(None, norm, existing).ratio() >= threshold:
            return True
    return False


def dedupe_questions(questions):
    unique = []
    seen_normalized = []
    for q in questions:
        if not isinstance(q, dict) or "question" not in q or "options" not in q:
            print(f"  [dedupe] Skipping malformed item: {q!r}")
            continue
        if not q["question"] or not isinstance(q["options"], list) or len(q["options"]) < 2:
            print(f"  [dedupe] Skipping incomplete question: {q!r}")
            continue
        if is_near_duplicate(q["question"], seen_normalized):
            print(f"  [dedupe] Duplicate skipped: {q['question']}")
            continue
        unique.append(q)
        seen_normalized.append(normalize(q["question"]))
    return unique


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_pdf(pdf_path, model_name):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{base_name}.json"

    final_output = {
        "chapter": base_name.replace("-", " ").replace("_", " ").title(),
        "class": "Class-VI",
        "language": "English",
        "year": "2025",
        "subject": "Social Science",
        "questions": [],
    }

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        pages_text = [page.extract_text() or "" for page in reader.pages]

    print(f"\nProcessing '{pdf_path}' ({len(pages_text)} pages) using {model_name}...\n")

    chunks = build_page_chunks(pages_text)
    print(f"Processing {len(chunks)} page(s) individually.\n")

    all_raw_questions = []

    for i, chunk in enumerate(chunks):
        page_num = chunk["pages"][0]
        print(f"==================== Page {page_num}/{len(pages_text)} ====================")

        print("Extracting important facts...")
        facts = extract_important_facts(chunk["text"], model_name)
        print(f"  -> {len(facts)} facts found.")

        fact_questions = []
        if facts:
            print("Generating one question per fact...")
            fact_questions = generate_questions_from_facts(facts, chunk["text"], model_name)
            print(f"  -> {len(fact_questions)} questions generated from facts.")

        print("Running general coverage pass...")
        general_questions = generate_general_questions(chunk["text"], model_name)
        print(f"  -> {len(general_questions)} additional questions generated.")

        all_raw_questions.extend(fact_questions)
        all_raw_questions.extend(general_questions)
        print()

    print("Deduplicating questions across the whole chapter...")
    unique_questions = dedupe_questions(all_raw_questions)
    print(f"  -> {len(unique_questions)} unique questions kept out of {len(all_raw_questions)} generated.\n")

    for idx, q in enumerate(unique_questions, start=1):
        final_output["questions"].append({
            "id": idx,
            "question": q.get("question", ""),
            "options": q.get("options", []),
            "correctAnswer": q.get("correctAnswer", 0),
            "explanation": q.get("explanation", ""),
        })

    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(final_output, outfile, indent=2, ensure_ascii=False)

    print(f"Completed! Total unique questions: {len(final_output['questions'])}")
    print(f"Saved to: {output_filename}")


if __name__ == "__main__":
    pdf_input = input("Please enter the path to your PDF file: ").strip()

    if (pdf_input.startswith('"') and pdf_input.endswith('"')) or \
       (pdf_input.startswith("'") and pdf_input.endswith("'")):
        pdf_input = pdf_input[1:-1]

    OLLAMA_MODEL = "qwen2.5-coder:7b"

    if os.path.exists(pdf_input):
        process_pdf(pdf_input, OLLAMA_MODEL)
    else:
        print(f"Error: Could not find file at '{pdf_input}'.")