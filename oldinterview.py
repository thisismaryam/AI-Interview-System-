import os
import sys
import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from speech_io import speak, transcribe
from audio_record import record_until_enter

load_dotenv()


class InterviewState(TypedDict):
    max_questions: int
    current_question: str
    current_answer: str
    transcript: list[dict]
    relevant_answer: float
    done: bool
    jd_text: str
    resume_text: str
    context_summary: str
    overall_score: float
    feedback: str
    verdict: str


api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env file!")
    sys.exit(1)

llm = ChatGroq(model=model_name, groq_api_key=api_key)


def call_llm(prompt: str):
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return ' '.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content).strip()
    return content.strip()


def load_text_file(filename: str):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read().strip()


def ask_question(state: InterviewState):
    if len(state["transcript"]) == 0:
        question = "Hello, let's start with your introduction. Tell me about your experience with this role."
    else:
        history = ""
        for turn in state["transcript"][-3:]:
            history += "Q: " + turn["question"] + "\n"
            history += "A: " + turn["answer"] + "\n"

        prompt = f"""
You are a technical interviewer conducting a job interview.
CONTEXT SUMMARY (JD + resume match):
{state['context_summary']}

CONVERSATION HISTORY:
{history}

IMPORTANT INSTRUCTIONS: YOU MUST FOLLOW THESE:
1. FIRST: Review the candidate's RESUME carefully - note their claimed skills, experience, projects, and technologies
2. Compare the candidate's previous answers against their RESUME
3. If the candidate's answers contradict what's in their RESUME, politely ask about the conflict.
4. Ask questions that explore the specific technologies and experience mentioned in their RESUME
5. Use the JOB DESCRIPTION to identify required skills, then cross-reference with their RESUME
6. Ask follow-up questions based on both their RESUME content AND their previous answers
7. DO NOT ask generic questions - make questions specific to their background
8. If the candidate mentions using a specific tool, IDE, library, or AI assistant (e.g. Cursor,
   Copilot, VS Code), do NOT ask them to describe, justify, or elaborate on that tool itself -
   that's not what's being evaluated. Instead ask about the underlying problem they solved, the
   decisions they made, or what they learned. The tool name is incidental, not the topic.

Based on the conversation, resume, and job description, ask the next interview question.
Return ONLY the question, no extra text.
"""
        question = call_llm(prompt)

    state["current_question"] = question
    print(f"\nInterviewer: {question}\n")
    speak(question)
    return state


def analyze_answer(state: InterviewState):
    audio_path = record_until_enter(
        f"turn_{len(state['transcript']) + 1}_answer.wav")
    answer = transcribe(audio_path).strip()
    print(f"You (transcribed): {answer}")

    state["current_answer"] = answer
    return state


def check_relevance(question: str, answer: str):
    prompt = f"""
Question: {question}
Answer: {answer}

On a scale of 0 to 10, where:
- 0 means completely irrelevant/off-topic
- 5 means partially relevant
- 10 means perfectly relevant and comprehensive

Rate how relevant this answer is to the question. Consider:
1. Does it directly address the question?
2. Does it provide specific examples?
3. Is it comprehensive enough?
4. Does it show understanding of the topic?

Return ONLY a number between 0 and 10, no other text.
"""
    try:
        return float(call_llm(prompt))
    except (ValueError, TypeError):
        return 5.0


def evaluate(state: InterviewState):
    relevance_score = check_relevance(
        state["current_question"], state["current_answer"])
    state["relevant_answer"] = relevance_score
    state["transcript"].append({
        "question": state["current_question"],
        "answer": state["current_answer"],
        "score": relevance_score,
    })

    if relevance_score < 3:
        print(
            f"\n[System: Your answer was not very relevant to the question. Score: {relevance_score:.1f}/10]")
    elif relevance_score < 6:
        print(
            f"\n[System: Your answer was partially relevant to the question. Score: {relevance_score:.1f}/10]")
    else:
        print(f"\n[System: Relevant answer! Score: {relevance_score:.1f}/10]")

    # HARD CAP: transcript grows by exactly one entry per turn (whether it
    # was a follow-up or a fresh topic - ask_question doesn't distinguish),
    # so this length check alone guarantees we never exceed max_questions.
    if len(state["transcript"]) >= state["max_questions"]:
        state["done"] = True
        print("\nInterview complete.")
    return state


def finalize(state: InterviewState):
    report_text = "=== INTERVIEW REPORT ===\n\n"
    for turn in state["transcript"]:
        report_text += f"Q: {turn['question']}\nA: {turn['answer']}\nScore: {turn['score']}\n\n"

    scores = [turn["score"] for turn in state["transcript"]]
    avg_score = sum(scores) / len(scores) if scores else 0
    report_text += f"Average Score: {avg_score:.1f}/10\n\n"

    prompt = f"""
Based on this interview transcript, average relevance score of {avg_score:.1f}/10,
and the job description below, decide if the candidate is qualified.

JOB DESCRIPTION:
{state['jd_text']}

Reply in this exact format:
Result: <Passed / Partially Passed / Failed>
Reason: <one line reason>
"""
    verdict_text = call_llm(prompt)
    report_text += "--- FINAL RESULT ---\n" + verdict_text + "\n"

    with open("interview_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Parse the verdict so app.py can hand it to the frontend directly
    # instead of the frontend having to parse raw LLM text.
    result_match = re.search(r'Result:\s*(.+)', verdict_text)
    reason_match = re.search(r'Reason:\s*(.+)', verdict_text)

    state["overall_score"] = avg_score
    state["verdict"] = result_match.group(
        1).strip() if result_match else "Unknown"
    state["feedback"] = reason_match.group(
        1).strip() if reason_match else verdict_text

    print("\nInterview submitted successfully.")
    print("Thank you for completing the interview!")
    return state


def route_after_evaluate(state: InterviewState):
    return "finalize" if state["done"] else "ask_question"


graph = StateGraph(InterviewState)
graph.add_node("ask_question", ask_question)
graph.add_node("analyze_answer", analyze_answer)
graph.add_node("evaluate", evaluate)
graph.add_node("finalize", finalize)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "analyze_answer")
graph.add_edge("analyze_answer", "evaluate")
graph.add_conditional_edges("evaluate", route_after_evaluate, {
    "ask_question": "ask_question",
    "finalize": "finalize",
})
graph.add_edge("finalize", END)

graph_app = graph.compile()  # NOTE: was named "app" - renamed so it can
# never collide with Flask's app object in app.py.


def build_initial_state(resume_text: str = None, max_questions: int = 5):
    job_description = load_text_file("jd.txt")
    resume = resume_text if resume_text is not None else load_text_file(
        "resume.txt")
    summary_prompt = f"Summarize the key skills and experience match between this JD and resume in 5-6 lines:\n\nJD:\n{job_description}\n\nRESUME:\n{resume}"
    context_summary = call_llm(summary_prompt)

    return {
        "max_questions": max_questions,
        "current_question": "",
        "current_answer": "",
        "transcript": [],
        "relevant_answer": 0.0,
        "done": False,
        "jd_text": job_description,
        "resume_text": resume,
        "context_summary": context_summary,
        "overall_score": 0.0,
        "feedback": "",
        "verdict": "",
    }


def run_interview():
    initial_state = build_initial_state()
    final_state = graph_app.invoke(
        initial_state, config={"recursion_limit": 50})

    print("\n=== FINAL STATE ===")
    print(final_state)


if __name__ == "__main__":
    # Only render the graph diagram when running this file directly (it's
    # a debug aid, not something every Flask server boot should depend on).
    graph_app.get_graph().draw_mermaid_png(output_file_path="interview_graph.png")
    run_interview()
