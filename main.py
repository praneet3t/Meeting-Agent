import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from database import SessionLocal, Task
import os

# --- 1. CONFIGURATION & INPUT ---

# The path to your JSON transcript file.
# Using a raw string (r"...") is important to handle the backslashes in Windows paths correctly.
JSON_TRANSCRIPT_PATH = r"C:\Users\apran\Videos\Cin\LIBRARY\Meeting Agent\transcript.json"

# Specify the small, open-source model we'll use
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

# --- 2. DATA LOADING & AI PROCESSING ---

def load_transcript_from_json(file_path):
    """Loads a transcript from a JSON file and formats it into a single string."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Format the transcript into a readable string for the LLM
        formatted_transcript = f"Meeting Title: {data['meeting_title']}\n"
        formatted_transcript += f"Date: {data['date']}\n"
        formatted_transcript += f"Participants: {', '.join(data['participants'])}\n\n"
        
        for entry in data['transcript']:
            formatted_transcript += f"{entry['speaker']} ({entry['timestamp']}): {entry['dialogue']}\n"
            
        return formatted_transcript
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: The file at {file_path} is not a valid JSON.")
        return None
    except KeyError as e:
        print(f"❌ Error: The JSON is missing a required key: {e}")
        return None

def load_model():
    """Loads the tokenizer and model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # Use float16 for less memory
        device_map="auto" # Automatically use GPU if available
    )
    return tokenizer, model

def process_transcript(tokenizer, model, transcript):
    """Generates MoM and extracts tasks using the LLM."""
    prompt = f"""
    You are an expert meeting assistant. Analyze the following transcript.
    Your tasks are:
    1.  Generate a concise "Minutes of Meeting" summary.
    2.  Extract all action items into a structured JSON format.

    The JSON output must be a single object with a key "tasks", which is a list of objects.
    Each task object must have three keys: "task_description", "assignee", and "due_date".

    Respond with a single JSON object containing "minutes" and "tasks".

    Transcript:
    ---
    {transcript}
    ---
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that processes meeting transcripts."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    try:
        json_response = json.loads(response[response.find('{'):response.rfind('}')+1])
        return json_response
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from the model's response.")
        return None

def save_tasks_to_db(tasks):
    """Saves the extracted tasks into the SQLite database."""
    db = SessionLocal()
    try:
        for task_item in tasks:
            new_task = Task(
                description=task_item.get("task_description"),
                assignee=task_item.get("assignee"),
                due_date_str=task_item.get("due_date"),
                status="To Do"
            )
            db.add(new_task)
        db.commit()
        print(f"Successfully saved {len(tasks)} tasks to the database.")
    finally:
        db.close()

# --- 3. AGENT FUNCTIONS (TOOLS) ---

def get_all_tasks():
    """Agent Tool: Retrieves all tasks from the database."""
    db = SessionLocal()
    tasks = db.query(Task).all()
    db.close()
    if not tasks:
        return "No tasks found in the database."
    response = "Here are the current tasks:\n"
    for task in tasks:
        response += f"- Task: '{task.description}', Assignee: {task.assignee}, Due: {task.due_date_str}, Status: {task.status}\n"
    return response

def update_task_status(task_description, new_status):
    """Agent Tool: Updates the status of a specific task."""
    db = SessionLocal()
    task = db.query(Task).filter(Task.description.ilike(f"%{task_description}%")).first()
    if task:
        task.status = new_status
        db.commit()
        response = f"Updated status for task '{task.description}' to '{new_status}'."
    else:
        response = f"Could not find a task matching '{task_description}'."
    db.close()
    return response

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    print("🚀 Starting ActionFlow Terminal Agent...")
    
    # Load the transcript from the specified JSON file
    meeting_transcript = load_transcript_from_json(JSON_TRANSCRIPT_PATH)
    
    if meeting_transcript:
        print("✅ Transcript loaded successfully.")
        print("Loading AI model... (This might take a minute the first time)")
        tokenizer, model = load_model()
        
        print("\nProcessing transcript...")
        results = process_transcript(tokenizer, model, meeting_transcript)
        
        if results:
            # Display MoM and save tasks
            print("\n--- Minutes of Meeting ---")
            print(results.get("minutes", "No summary generated."))
            
            tasks = results.get("tasks", [])
            if tasks:
                print("\n--- Extracted Action Items ---")
                for task in tasks:
                    print(f"- {task}")
                save_tasks_to_db(tasks)
            else:
                print("\nNo action items were extracted.")
                
            # Start the interactive agent loop
            print("\n--- Task Agent is ready! ---")
            print("You can now ask about your tasks. Type 'quit' to exit.")
            
            while True:
                query = input("> ").lower().strip()
                if query == 'quit':
                    break
                
                # Simple rule-based agent logic
                if query.startswith("update status for"):
                    # Example: "update status for 'budget forecast' to 'In Progress'"
                    try:
                        desc = query.split("'")[1]
                        status = query.split("'")[3]
                        print(update_task_status(desc, status))
                    except IndexError:
                        print("Please use the format: update status for 'task description' to 'new status'")
                elif query in ["what are the tasks", "show all tasks", "list tasks"]:
                    print(get_all_tasks())
                else:
                    print("I can help with tasks. Try 'show all tasks' or 'update status for ... to ...'.")