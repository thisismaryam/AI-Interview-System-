"""
Interview System with Diverse Topics & End-of-Interview Scoring
- Covers multiple skills from JD and resume
- Follow-ups only when necessary
- Scoring ONLY at the end
"""

import os
import sys
import re
import random
from typing import TypedDict, List, Dict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from speech_io import speak, transcribe
from audio_record import record_until_enter

load_dotenv()

# ==================== Configuration ====================

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env file!")
    sys.exit(1)

llm = ChatGroq(model=model_name, groq_api_key=api_key)

# ==================== State ====================


class InterviewState(TypedDict):
    max_questions: int
    current_question: str
    current_answer: str
    done: bool
    resume_text: str
    jd_text: str
    context_summary: str
    transcript: List[Dict]
    question_count: int
    final_score: float
    feedback: str
    covered_topics: List[str]  # Track topics already asked
    follow_up_in_progress: bool
    #  Track if current question already had a follow-up
    follow_up_used: bool

# ==================== Helper Functions ====================


def call_llm(prompt: str, expect_number: bool = False) -> str:
    try:
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        response = llm.invoke(prompt)
        content = response.content.strip()
        if expect_number:
            numbers = re.findall(r'\d+(?:\.\d+)?', content)
            return numbers[0] if numbers else "5.0"
        return content if content else "Tell me more about your experience."
    except Exception as e:
        print(f" LLM Error: {e}")
        return "5.0" if expect_number else "Tell me more about your experience."


def load_file(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()

# ==================== Question Generation ====================


def generate_question(state: InterviewState, is_followup: bool = False) -> str:
    """Generate a question (new topic or follow-up)."""

    if is_followup:
        last = state["transcript"][-1] if state["transcript"] else None
        if not last:
            return "Could you tell me more about that?"

        prompt = f"""
You're a job interviewer. You asked:
Q: {last['question']}

Candidate answered:
A: {last['answer']}

Ask a brief follow-up question that:
1. Probes deeper into their answer
2. Asks for a specific example or clarification
3. Keeps it short and focused

Return ONLY the question, no extra text.
"""
        return call_llm(prompt)

    else:
        # NEW QUESTION - ask about a DIFFERENT topic
        prompt = f"""
You're a job interviewer for this role:

JOB DESCRIPTION:
{state['jd_text']}

CANDIDATE RESUME:
{state['resume_text'][:500]}...

Topics already covered: {state.get('covered_topics', []) or 'None yet'}

Ask a NEW question about a skill or experience NOT yet covered.
Focus on different aspects of the job description.
Be specific to the candidate's resume.
Keep it concise.

Return ONLY the question, no extra text.
"""
        return call_llm(prompt)


def extract_topic_from_question(question: str) -> str:
    """Extract the main topic from a question to track coverage."""
    prompt = f"Extract the main topic from this question in 2-3 words: {question}"
    return call_llm(prompt)

# ==================== Follow-up Decision ====================


def should_follow_up(state: InterviewState) -> bool:
    """Decide if we should ask a follow-up."""

    # No transcript? No follow-up.
    if not state["transcript"]:
        return False

    # No more questions left
    if state["question_count"] >= state["max_questions"]:
        return False

    # Already followed up on this question? Then move on.
    if state.get("follow_up_used", False):
        return False

    # If we already did a follow-up on the last question, don't do another.
    if state["transcript"][-1].get("follow_up", False):
        return False

    # Check last answer
    last_answer = state["transcript"][-1]["answer"]
    words = len(last_answer.split())

    # Only follow up if answer is very short (< 10 words) or contains "not sure", "don't know", etc.
    if words < 10:
        return True

    vague_indicators = ["not sure", "don't know",
                        "maybe", "i think", "i guess"]
    if any(indicator in last_answer.lower() for indicator in vague_indicators):
        return True

    return False

# ==================== LangGraph Nodes ====================


def ask_question(state: InterviewState):
    """Ask the next question (new or follow-up)."""

    # Check if we should follow up
    if should_follow_up(state):
        question = generate_question(state, is_followup=True)
        state["follow_up_in_progress"] = True
        # Mark that we used follow-up on this question
        state["follow_up_used"] = True
        print(
            f"Follow-up {state['question_count']+1}/{state['max_questions']}")
    else:
        # New question - mark follow-up used as False for next round
        state["follow_up_used"] = False
        if state["question_count"] == 0:
            question = "Hello, let's start with your introduction. Tell me about your experience with this role."
        else:
            question = generate_question(state, is_followup=False)
            # Extract topic and add to covered list
            topic = extract_topic_from_question(question)
            if topic and topic not in state.get("covered_topics", []):
                state["covered_topics"].append(topic)
        state["follow_up_in_progress"] = False
        print(
            f" New question {state['question_count']+1}/{state['max_questions']}")

    # Fallback if question is empty
    if not question or question.strip() == "":
        question = "Could you tell me more about your experience?"

    state["current_question"] = question

    # Speak the question
    print(f"\nInterviewer: {question}\n")
    speak(question)

    return state


def analyze_answer(state: InterviewState):
    """Record and transcribe the candidate's answer."""
    audio_path = record_until_enter(
        f"turn_{state['question_count'] + 1}_answer.wav")
    answer = transcribe(audio_path).strip()
    print(f"You: {answer}")

    state["current_answer"] = answer
    return state


def evaluate(state: InterviewState):
    """Store answer and increment question count (NO SCORING)."""

    # Store the exchange
    state["transcript"].append({
        "question": state["current_question"],
        "answer": state["current_answer"],
        "follow_up": state.get("follow_up_in_progress", False)
    })

    # Increment question count
    state["question_count"] += 1
    print(
        f" Question {state['question_count']}/{state['max_questions']} completed")

    # Check if we're done
    if state["question_count"] >= state["max_questions"]:
        state["done"] = True
        print(" All questions completed!")

    return state


def finalize(state: InterviewState):
    """Score EVERYTHING at the end."""

    print("\n Evaluating entire interview...")

    # Build full conversation
    conv = ""
    for i, t in enumerate(state["transcript"], 1):
        follow = " (Follow-up)" if t.get("follow_up", False) else ""
        conv += f"Q{i}{follow}: {t['question']}\n"
        conv += f"A{i}: {t['answer']}\n\n"

    # Get overall score and feedback
    overall_score, feedback = score_overall(
        conv, state["resume_text"], state["jd_text"])

    # Store results
    state["final_score"] = overall_score
    state["feedback"] = feedback

    # Add score to each transcript entry
    for t in state["transcript"]:
        t["score"] = overall_score

    # Generate report
    generate_report(state)

    print(f"\n Final Score: {overall_score:.1f}/10")
    print(f" Feedback: {feedback}")

    return state


def score_overall(conversation: str, resume: str, jd: str) -> tuple:
    """Ask LLM to score the entire interview."""
    prompt = f"""
Evaluate this COMPLETE job interview.

RESUME: {resume[:300]}...
JD: {jd[:300]}...

CONVERSATION:
{conversation}

Rate on (0-10):
1. Technical competence / Job fit
2. Communication
3. Problem-solving
4. Growth during the interview

Also provide 2-3 sentences of constructive feedback.

Format exactly:
Score: <number>
Feedback: <feedback>
"""
    response = call_llm(prompt)

    score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response)
    score = float(score_match.group(1)) if score_match else 5.0

    feedback_match = re.search(r'Feedback:\s*(.+?)(?=$)', response, re.DOTALL)
    feedback = feedback_match.group(1).strip(
    ) if feedback_match else "Good interview."

    return min(10, max(0, score)), feedback


def generate_report(state: InterviewState):
    """Save interview report to file."""
    report = ["=" * 50, "INTERVIEW REPORT", "=" * 50, ""]
    report.append(f"Questions Asked: {len(state['transcript'])}")
    report.append(f"Overall Score: {state['final_score']:.1f}/10")
    report.append("")
    report.append(f"Feedback: {state['feedback']}")
    report.append("")

    for i, t in enumerate(state["transcript"], 1):
        follow = " (Follow-up)" if t.get("follow_up", False) else ""
        report.append(f"\n--- Question {i}{follow} ---")
        report.append(f"Q: {t['question']}")
        report.append(f"A: {t['answer']}")
        report.append(f"Score: {t.get('score', 0):.1f}/10")

    with open("interview_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\nReport saved to interview_report.txt")

# ==================== Graph ====================


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

app = graph.compile()

# ==================== Build Initial State ====================


def build_initial_state(resume_text: str = None):
    jd = load_file("jd.txt")
    resume = resume_text if resume_text else load_file("resume.txt")

    prompt = f"Summarize the match between this JD and resume in 1-2 sentences:\n\nJD: {jd}\nRESUME: {resume}"
    summary = call_llm(prompt)

    return {
        "max_questions": 5,          # <<< CHANGE THIS to adjust total questions
        "current_question": "",
        "current_answer": "",
        "done": False,
        "resume_text": resume,
        "jd_text": jd,
        "context_summary": summary,
        "transcript": [],
        "question_count": 0,
        "final_score": 0.0,
        "feedback": "",
        "covered_topics": [],
        "follow_up_in_progress": False,
        "follow_up_used": False
    }

# ==================== Run Interview ====================


def run_interview():
    state = build_initial_state()
    final_state = app.invoke(state, config={"recursion_limit": 50})
    return final_state


if __name__ == "__main__":
    run_interview()
