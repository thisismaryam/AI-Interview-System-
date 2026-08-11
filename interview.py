"""
Interview System with Diverse Topics & Weighted Scoring
- Covers multiple skills from JD and resume
- Follow-ups only when necessary
- Weighted scoring across 10 dimensions
- Detailed justification for each score
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
    covered_topics: List[str]
    follow_up_in_progress: bool
    follow_up_used: bool
    current_audio: Optional[str]

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
        print(f"⚠️ LLM Error: {e}")
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
    if not state["transcript"]:
        return False
    if state["question_count"] >= state["max_questions"]:
        return False
    if state.get("follow_up_used", False):
        return False
    if state["transcript"][-1].get("follow_up", False):
        return False

    last_answer = state["transcript"][-1]["answer"]
    words = len(last_answer.split())

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

    if should_follow_up(state):
        question = generate_question(state, is_followup=True)
        state["follow_up_in_progress"] = True
        state["follow_up_used"] = True
        print(
            f"🔄 Follow-up {state['question_count']+1}/{state['max_questions']}")
    else:
        state["follow_up_used"] = False
        if state["question_count"] == 0:
            question = "Thank you for joining us today! To start, could you walk me through your background and how your experience aligns with this role?"
        else:
            question = generate_question(state, is_followup=False)
            topic = extract_topic_from_question(question)
            if topic and topic not in state.get("covered_topics", []):
                state["covered_topics"].append(topic)
        state["follow_up_in_progress"] = False
        print(
            f"📝 New question {state['question_count']+1}/{state['max_questions']}")

    if not question or question.strip() == "":
        question = "Could you tell me more about your experience?"

    state["current_question"] = question
    print(f"\nInterviewer: {question}\n")

    # Generate TTS audio (for cloud deployment)
    from speech_io import speak
    audio_file = speak(question)
    state["current_audio"] = audio_file

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

    state["transcript"].append({
        "question": state["current_question"],
        "answer": state["current_answer"],
        "follow_up": state.get("follow_up_in_progress", False)
    })

    state["question_count"] += 1
    print(
        f"✅ Question {state['question_count']}/{state['max_questions']} completed")

    if state["question_count"] >= state["max_questions"]:
        state["done"] = True
        print("🎯 All questions completed!")

    return state


def finalize(state: InterviewState):
    """Score EVERYTHING at the end with detailed justification and weighted dimensions."""

    print("\n📊 Evaluating entire interview...")

    scored_transcript = []
    total_weighted_score = 0
    max_possible_total = 0

    for i, t in enumerate(state["transcript"], 1):
        question_score, detailed_scores, justification = score_question(
            t["question"],
            t["answer"],
            state["resume_text"],
            state["jd_text"],
            i
        )

        t["question_score"] = question_score
        t["detailed_scores"] = detailed_scores
        t["justification"] = justification
        scored_transcript.append(t)

        total_weighted_score += question_score["overall"]
        max_possible_total += 100

    overall_percentage = (
        total_weighted_score / max_possible_total) * 100 if max_possible_total > 0 else 0

    overall_feedback = generate_overall_feedback(
        scored_transcript,
        state["resume_text"],
        state["jd_text"]
    )

    state["final_score"] = round(overall_percentage, 1)
    state["feedback"] = overall_feedback
    state["transcript"] = scored_transcript

    generate_report(state)

    print(f"\n📊 Final Score: {overall_percentage:.1f}/100")
    print(f"📝 Feedback: {overall_feedback}")

    return state


def score_question(question: str, answer: str, resume: str, jd: str, q_num: int) -> tuple:
    """
    Score a single question across weighted dimensions with justification.
    Returns: (weighted_score_dict, detailed_scores_dict, justification_text)
    """

    prompt = f"""
You are evaluating a job interview response. Score this answer across 10 dimensions with specific weights.

**RESUME (candidate's background):**
{resume[:500]}...

**JOB DESCRIPTION (what we're hiring for):**
{jd[:500]}...

**QUESTION {q_num}:**
{question}

**CANDIDATE'S ANSWER:**
{answer}

---

**SCORING DIMENSIONS AND WEIGHTS:**

1. **Technical Knowledge** (Weight: 15%)
   - Understanding of technologies, concepts, tools, and frameworks required for the role
   - Correct use of technical terminology
   - Depth of technical understanding demonstrated

2. **Problem-Solving Skills** (Weight: 15%)
   - Ability to analyze a problem logically
   - Ability to reason through challenges
   - Reaches practical, implementable solutions
   - Shows structured approach to problem-solving

3. **Communication Skills** (Weight: 10%)
   - Clarity of expression
   - Ability to articulate complex ideas simply
   - Active listening and responsiveness to the question
   - Professional language and tone

4. **Role Understanding** (Weight: 10%)
   - Demonstrates understanding of what the role entails
   - Aligns answers with job requirements
   - Shows awareness of industry context
   - Understands how their skills fit the role

5. **Relevant Experience** (Weight: 10%)
   - Provides concrete examples from past experience
   - Examples are directly relevant to the question
   - Demonstrates depth of experience
   - Shows application of skills in real scenarios

6. **Critical Thinking** (Weight: 10%)
   - Evaluates situations from multiple perspectives
   - Questions assumptions appropriately
   - Shows analytical depth
   - Demonstrates sound judgment

7. **Adaptability & Learning** (Weight: 10%)
   - Shows willingness to learn new things
   - Demonstrates flexibility in approach
   - Handles uncertainty or ambiguity well
   - Shows growth mindset

8. **Behavioral Skills** (Weight: 10%)
   - Self-awareness and reflection
   - Ability to work in teams
   - Handles feedback and challenges professionally
   - Shows initiative and ownership

9. **Professionalism & Confidence** (Weight: 5%)
   - Professional attitude and composure
   - Confidence in responses
   - Appropriate behavior during the interview
   - Maintains professional demeanor

10. **Answer Relevance & Quality** (Weight: 5%)
    - Directly addresses the question asked
    - Not vague, unrelated, or overly generalized
    - Provides specific, concrete information
    - Well-structured and organized response

---

**OUTPUT FORMAT (MUST FOLLOW EXACTLY):**

For each dimension, provide a score (0-100) and a brief justification.

Technical Knowledge: <score>
Justification: <brief reason>

Problem-Solving Skills: <score>
Justification: <brief reason>

Communication Skills: <score>
Justification: <brief reason>

Role Understanding: <score>
Justification: <brief reason>

Relevant Experience: <score>
Justification: <brief reason>

Critical Thinking: <score>
Justification: <brief reason>

Adaptability & Learning: <score>
Justification: <brief reason>

Behavioral Skills: <score>
Justification: <brief reason>

Professionalism & Confidence: <score>
Justification: <brief reason>

Answer Relevance & Quality: <score>
Justification: <brief reason>

Weighted Overall Score: <calculated score out of 100 using the weights above>
Overall Justification: <1-2 sentences summarizing the overall performance on this question>
"""

    response = call_llm(prompt)

    # Define weights
    weights = {
        "technical": 0.15,
        "problem_solving": 0.15,
        "communication": 0.10,
        "role_understanding": 0.10,
        "relevant_experience": 0.10,
        "critical_thinking": 0.10,
        "adaptability": 0.10,
        "behavioral": 0.10,
        "professionalism": 0.05,
        "relevance": 0.05
    }

    dimension_patterns = {
        "technical": r"Technical Knowledge:\s*(\d+)",
        "problem_solving": r"Problem-Solving Skills:\s*(\d+)",
        "communication": r"Communication Skills:\s*(\d+)",
        "role_understanding": r"Role Understanding:\s*(\d+)",
        "relevant_experience": r"Relevant Experience:\s*(\d+)",
        "critical_thinking": r"Critical Thinking:\s*(\d+)",
        "adaptability": r"Adaptability & Learning:\s*(\d+)",
        "behavioral": r"Behavioral Skills:\s*(\d+)",
        "professionalism": r"Professionalism & Confidence:\s*(\d+)",
        "relevance": r"Answer Relevance & Quality:\s*(\d+)"
    }

    justification_patterns = {
        "technical": r"Technical Knowledge:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "problem_solving": r"Problem-Solving Skills:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "communication": r"Communication Skills:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "role_understanding": r"Role Understanding:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "relevant_experience": r"Relevant Experience:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "critical_thinking": r"Critical Thinking:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "adaptability": r"Adaptability & Learning:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "behavioral": r"Behavioral Skills:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "professionalism": r"Professionalism & Confidence:.*?Justification:\s*(.+?)(?=\n\w+:|$)",
        "relevance": r"Answer Relevance & Quality:.*?Justification:\s*(.+?)(?=\n\w+:|$)"
    }

    # Extract scores
    scores = {}
    for key, pattern in dimension_patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        scores[key] = int(match.group(1)) if match else 70

    # Extract justifications
    justifications = {}
    for key, pattern in justification_patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        justifications[key] = match.group(1).strip(
        ) if match else "No justification provided."

    # Calculate weighted score
    weighted_score = 0
    for key, weight in weights.items():
        weighted_score += scores.get(key, 70) * weight

    weighted_score = round(weighted_score, 1)

    # Extract overall justification
    overall_justif_pattern = r"Overall Justification:\s*(.+?)(?=$)"
    overall_match = re.search(overall_justif_pattern, response, re.DOTALL)
    overall_justification = overall_match.group(
        1).strip() if overall_match else "Good response."

    # Build detailed scores dictionary with human-readable names
    detailed_scores = {
        "Technical Knowledge": scores["technical"],
        "Problem-Solving Skills": scores["problem_solving"],
        "Communication Skills": scores["communication"],
        "Role Understanding": scores["role_understanding"],
        "Relevant Experience": scores["relevant_experience"],
        "Critical Thinking": scores["critical_thinking"],
        "Adaptability & Learning": scores["adaptability"],
        "Behavioral Skills": scores["behavioral"],
        "Professionalism & Confidence": scores["professionalism"],
        "Answer Relevance & Quality": scores["relevance"]
    }

    return {
        "overall": weighted_score,
        "dimensions": scores
    }, detailed_scores, {
        "dimensions": justifications,
        "overall": overall_justification
    }


def generate_overall_feedback(transcript: List[Dict], resume: str, jd: str) -> str:
    """Generate overall feedback based on all question scores."""

    dim_scores = {
        "Technical Knowledge": [],
        "Problem-Solving Skills": [],
        "Communication Skills": [],
        "Role Understanding": [],
        "Relevant Experience": [],
        "Critical Thinking": [],
        "Adaptability & Learning": [],
        "Behavioral Skills": [],
        "Professionalism & Confidence": [],
        "Answer Relevance & Quality": []
    }

    for t in transcript:
        if "detailed_scores" in t:
            for dim, score in t["detailed_scores"].items():
                if dim in dim_scores:
                    dim_scores[dim].append(score)

    avg_scores = {}
    for dim, scores in dim_scores.items():
        avg_scores[dim] = sum(scores) / len(scores) if scores else 0

    strengths = []
    weaknesses = []

    priority_dims = ["Technical Knowledge", "Problem-Solving Skills"]

    for dim, score in avg_scores.items():
        if score >= 80:
            if dim in priority_dims:
                strengths.append(f"⭐ {dim}: Excellent ({score:.0f}/100)")
            else:
                strengths.append(f"✓ {dim}: Good ({score:.0f}/100)")
        elif score < 60:
            if dim in priority_dims:
                weaknesses.append(
                    f"⚠️ {dim}: Needs significant improvement ({score:.0f}/100)")
            else:
                weaknesses.append(
                    f"○ {dim}: Could be improved ({score:.0f}/100)")

    feedback = f"Overall Performance: {avg_scores.get('Technical Knowledge', 0):.1f}/100\n\n"

    if strengths:
        feedback += "✅ STRENGTHS:\n" + \
            "\n".join([f"  • {s}" for s in strengths]) + "\n\n"

    if weaknesses:
        feedback += "⚠️ AREAS FOR IMPROVEMENT:\n" + \
            "\n".join([f"  • {w}" for w in weaknesses]) + "\n\n"

    recommendations = []
    sorted_dims = sorted(avg_scores.items(), key=lambda x: x[1])
    lowest_3 = sorted_dims[:3]

    for dim, score in lowest_3:
        if score < 70:
            if "Technical" in dim:
                recommendations.append(
                    "Review core technical concepts and practice explaining them clearly")
            elif "Problem" in dim:
                recommendations.append(
                    "Practice breaking down problems systematically before answering")
            elif "Communication" in dim:
                recommendations.append(
                    "Structure answers using the STAR method (Situation, Task, Action, Result)")
            elif "Role" in dim:
                recommendations.append(
                    "Study the job description thoroughly and align answers to specific requirements")
            elif "Experience" in dim:
                recommendations.append(
                    "Prepare concrete examples from past work or projects")
            elif "Critical" in dim:
                recommendations.append(
                    "Practice analyzing scenarios from multiple perspectives")
            elif "Adaptability" in dim:
                recommendations.append(
                    "Share examples of learning new skills or adapting to change")
            elif "Behavioral" in dim:
                recommendations.append(
                    "Prepare stories demonstrating teamwork, leadership, and initiative")
            elif "Professionalism" in dim:
                recommendations.append(
                    "Practice maintaining professional tone and confidence")
            elif "Relevance" in dim:
                recommendations.append(
                    "Listen carefully to questions and ensure answers directly address them")

    if recommendations:
        feedback += "💡 RECOMMENDATIONS:\n" + \
            "\n".join([f"  • {r}" for r in recommendations[:5]]) + "\n\n"

    overall_score = sum(avg_scores.values()) / \
        len(avg_scores) if avg_scores else 0
    feedback += f"📊 Average Score Across All Dimensions: {overall_score:.1f}/100\n"

    if overall_score >= 85:
        feedback += "🏆 Overall Assessment: Outstanding performance. Strong fit for the role."
    elif overall_score >= 70:
        feedback += "✅ Overall Assessment: Good performance. Solid candidate with some areas for growth."
    elif overall_score >= 55:
        feedback += "📈 Overall Assessment: Adequate performance. Candidate shows potential but needs development."
    else:
        feedback += "📚 Overall Assessment: Needs significant improvement in multiple areas to be competitive for this role."

    return feedback


def generate_report(state: InterviewState):
    """Save detailed interview report with per-question justification."""
    report = ["=" * 80, "DETAILED INTERVIEW REPORT", "=" * 80, ""]
    report.append(f"Questions Asked: {len(state['transcript'])}")
    report.append(f"Overall Score: {state['final_score']:.1f}/100")
    report.append("")

    report.append("=" * 80)
    report.append("SCORING WEIGHTS LEGEND")
    report.append("=" * 80)
    report.append("  • Technical Knowledge: 15%")
    report.append("  • Problem-Solving Skills: 15%")
    report.append("  • Communication Skills: 10%")
    report.append("  • Role Understanding: 10%")
    report.append("  • Relevant Experience: 10%")
    report.append("  • Critical Thinking: 10%")
    report.append("  • Adaptability & Learning: 10%")
    report.append("  • Behavioral Skills: 10%")
    report.append("  • Professionalism & Confidence: 5%")
    report.append("  • Answer Relevance & Quality: 5%")
    report.append("")

    report.append("=" * 80)
    report.append("PER-QUESTION BREAKDOWN")
    report.append("=" * 80)
    report.append("")

    for i, t in enumerate(state["transcript"], 1):
        follow = " (Follow-up)" if t.get("follow_up", False) else ""
        report.append(f"\n--- Question {i}{follow} ---")
        report.append(f"Q: {t['question']}")
        report.append(f"A: {t['answer']}")

        if "detailed_scores" in t and "justification" in t:
            report.append("")
            report.append("📊 SCORE BREAKDOWN:")

            for dim, score in t["detailed_scores"].items():
                emoji = "✅" if score >= 80 else "📈" if score >= 60 else "⚠️"
                report.append(f"  {emoji} {dim}: {score}/100")

            report.append("")
            report.append(
                f"📝 Weighted Overall: {t['question_score']['overall']:.1f}/100")
            report.append("")

            report.append("📝 JUSTIFICATION:")
            for dim, justification in t["justification"]["dimensions"].items():
                dim_name = " ".join(word.capitalize()
                                    for word in dim.split("_"))
                report.append(f"  • {dim_name}: {justification}")

            report.append("")
            report.append(f"  • Overall: {t['justification']['overall']}")

        report.append("-" * 60)

    report.append("")
    report.append("=" * 80)
    report.append("OVERALL FEEDBACK")
    report.append("=" * 80)
    report.append("")
    report.append(state.get("feedback", "No feedback provided."))
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    with open("interview_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n📄 Detailed report saved to interview_report.txt")


def route_after_evaluate(state: InterviewState):
    return "finalize" if state["done"] else "ask_question"


# ==================== Graph ====================

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
        "max_questions": 5,
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
        "follow_up_used": False,
        "current_audio": None
    }


# ==================== Run Interview ====================

def run_interview():
    state = build_initial_state()
    final_state = app.invoke(state, config={"recursion_limit": 50})
    return final_state


if __name__ == "__main__":
    run_interview()
