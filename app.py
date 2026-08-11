import os
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from interview import build_initial_state, ask_question, evaluate, finalize
from speech_io import transcribe
from audio_record import start_recording, stop_recording
from resume_parser import extract_resume_text

app = Flask(__name__)

# Store state globally (simple for single-user testing)
state = {}
resume_text = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    global resume_text

    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    allowed_ext = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # Save file
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", filename)
    file.save(save_path)

    try:
        # Parse resume
        resume_text = extract_resume_text(save_path)
        print(f" Resume parsed: {len(resume_text)} characters")
        return jsonify({"status": "ok", "message": "Resume uploaded successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 400
    finally:
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/start", methods=["POST"])
def start():
    global state, resume_text

    if not resume_text:
        return jsonify({"error": "Upload a resume first."}), 400

    print(" Building initial state...")
    state = build_initial_state(resume_text=resume_text)

    print(" Asking first question...")
    state = ask_question(state)

    print(f" First question: {state['current_question'][:50]}...")
    return jsonify({"question": state["current_question"], "done": False})


@app.route("/start-recording", methods=["POST"])
def start_recording_route():
    try:
        start_recording()
        return jsonify({"status": "recording"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stop-recording", methods=["POST"])
def stop_recording_route():
    global state

    if not state:
        return jsonify({"error": "No active interview"}), 400

    try:
        # Stop recording and transcribe
        audio_path = stop_recording()
        answer = transcribe(audio_path).strip()
        print(f" Transcribed: {answer}")

        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Process answer
        return _advance(answer)
    except Exception as e:
        print(f" Error in stop_recording: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/skip-question", methods=["POST"])
def skip_question_route():
    global state

    if not state:
        return jsonify({"error": "No active interview"}), 400

    print("Skipping question (no response)")
    return _advance("")


def _advance(answer):
    global state

    try:
        # Set answer
        state["current_answer"] = answer

        # Evaluate
        print("Evaluating answer...")
        state = evaluate(state)
        if state["done"]:
            print("Interview complete, finalizing...")
            # <-- this adds "score" to each transcript entry
            state = finalize(state)
            return jsonify({
                "done": True,
                "overall_score": state["final_score"],   # <-- corrected key
                "feedback": state["feedback"],
                # <-- each entry has question, answer, score
                "transcript": state["transcript"],
            })
        # Ask next question
        print(" Asking next question...")
        state = ask_question(state)
        return jsonify({"question": state["current_question"], "done": False})

    except Exception as e:
        print(f" Error in _advance: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    print("Starting Flask server...")
    app.run(debug=True)
