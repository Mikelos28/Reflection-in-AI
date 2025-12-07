from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import re
import tempfile
import subprocess
from datasets import load_dataset
import os
import json
import matplotlib.pyplot as plt

STATS_FILE = "stats.json" #JSON FILE


# Create stats.json if missing
if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)


# JSON Logging
def save_steps(problem_id, initial=None, reflection=None, final=None, solved_step=None):
    """Append or update a record in stats.json for this problem."""
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False

    for entry in data:
        if entry["problem_id"] == problem_id:
            if initial is not None:
                entry["initial_solution"] = initial.strip()
            if reflection is not None:
                entry["reflection"] = reflection.strip()
            if final is not None:
                entry["final_solution"] = final.strip()
            if solved_step is not None:
                entry["solved_step"] = solved_step

            updated = True
            break

    if not updated:
        data.append({
            "problem_id": problem_id,
            "initial_solution": initial or "",
            "reflection": reflection or "",
            "final_solution": final or "",
            "solved_step": solved_step
        })

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

#PLOT FUNCTION
def plot_step_changes(json_path="stats.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

# ALL PLOT "COLUMNS"
    solved_without_reflection = 0
    solved_after_one_reflection = 0
    solved_after_loop = 0
    not_solved = 0

    for entry in data:
        step = entry.get("solved_step", None)

        if step == 1:
            solved_without_reflection += 1
        elif step == 2:
            solved_after_one_reflection += 1
        elif step == 3:
            solved_after_loop += 1
        else:
            not_solved += 1

    labels = [
        "Solved w/o reflection",
        "Solved w/ one reflection",
        "Solved in reflection loop",
        "Not solved"
    ]
    values = [
        solved_without_reflection,
        solved_after_one_reflection,
        solved_after_loop,
        not_solved
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Number of Problems Solved per Step")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    print("\n=== Summary ===")
    print("Solved without reflection:      ", solved_without_reflection)
    print("Solved with one reflection:     ", solved_after_one_reflection)
    print("Solved in reflection loop:      ", solved_after_loop)
    print("Not solved:                     ", not_solved)


dataset = load_dataset("openai/openai_humaneval", split="test") #Used dataset

model = OllamaLLM(model="gemma3") #Model used
embeddings = OllamaEmbeddings(model="mxbai-embed-large") #Embeddings used
vectorstore = Chroma(collection_name="llm_memory", embedding_function=embeddings, persist_directory="./chroma_db") #Vector Database

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)


# Prompt templates
template1 = """
You are a programmer. You will be given a series of programming problems
to solve and you will have to generate the code needed to solve the problem.
Try to think about it step by step before concluding and generating the code

User question:
{question}

Relevant past answers:
{reference}

"""
template2 = """
Follow these steps:
Step 1: Imagine you are reviewing this code as a senior engineer.
    - What would you critique or suggest improving?
    - Are there bugs, inefficiencies, or design flaws?
    - Use the test output and errors to locate the bug(s).

Step 2: Write down your suggestions for improvement and propose a corrected code snippet.

Here is the code the LLM generated before:
{former_code}

Test stdout:
{stdout}

Test stderr:
{stderr}

"""

template3 = """
You are a programmer expert in solving programming problems. You will be given a series of programming problems
to solve and you will have to generate the code needed to solve the problem. Try to think about it step by step
and explain what you're doing before concluding by generating the code.

User question:
{question}
"""

generative_prompt = ChatPromptTemplate.from_template(template1)
generative_chain = generative_prompt | model
reflective_prompt = ChatPromptTemplate.from_template(template2)
reflective_chain = reflective_prompt | model
no_reflection_prompt = ChatPromptTemplate.from_template(template3)
no_reflection_chain = no_reflection_prompt | model


# Extract code from the LLM answers
def extract_code(text):
    code = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if code: return code[0].strip()

    code = re.findall(r"```(.*?)```", text, re.DOTALL)
    if code: return code[0].strip()

    return text.strip()

# Testing the llm ansers
def run_tests(solution_code, test_code, timeout_seconds=5):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        fname = f.name
        f.write(solution_code)
        f.write("\n\n")
        f.write(test_code)
        f.flush()

    try:
        result = subprocess.run(
            ["python3", fname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=True
        )
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": "",
            "stderr": f"TimeoutExpired (>{timeout_seconds}s)",
            "returncode": None
        }

    finally:
        try:
            os.remove(fname)
        except:
            pass

#NO REFLECTION (step 1)

false_tasks = []
memo = {}

for task in dataset:
    task_id = task["task_id"]
    question = task["prompt"]
    test_code = task["test"]

    print(f"\nSolving {task_id} — No reflection") #Current problem

    llm_raw = no_reflection_chain.invoke({"question": question}) #Llm's initial response
    init_code = extract_code(llm_raw)

    test_result = run_tests(init_code, test_code) #Running the test

    if test_result["passed"]:
        save_steps(task_id, initial=llm_raw, solved_step=1)
    else:
        false_tasks.append(task_id)
        memo[task_id] = {
            "answer": llm_raw,
            "solution_code": init_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"]
        }
        save_steps(task_id, initial=llm_raw)

# ONE REFLECTION (step 2)

memo_reflection = {}

for task in dataset:
    task_id = task["task_id"]
    if task_id not in false_tasks:
        continue

    print(f"\nSolving {task_id} — STEP 2 (Reflection)")

    entry = memo[task_id]
    prev_code = entry["solution_code"]

    # reflection
    reflection = reflective_chain.invoke({
        "former_code": prev_code,
        "stdout": entry["stdout"],
        "stderr": entry["stderr"]
    })

    save_steps(task_id, reflection=reflection)

    # new solution
    improved_raw = generative_chain.invoke({"question": task["prompt"], "reference": reflection})
    improved_code = extract_code(improved_raw)

    test_result = run_tests(improved_code, task["test"])

    if test_result["passed"]:
        save_steps(task_id, final=improved_raw, solved_step=2)
        false_tasks.remove(task_id)
    else:
        memo_reflection[task_id] = improved_raw

# REFLECTION LOOP (step 3)

for task in dataset:
    task_id = task["task_id"]
    if task_id not in false_tasks:
        continue

    print(f"\nSolving {task_id} — STEP 3 (Looped Reflection)")

    curr_raw = memo_reflection.get(task_id, memo[task_id]["answer"])
    curr_code = extract_code(curr_raw)

    solved = False

    for z in range(10):
        test_result = run_tests(curr_code, task["test"])

        if test_result["passed"]:
            save_steps(task_id, final=curr_raw, solved_step=3)
            false_tasks.remove(task_id)
            solved = True
            break

        reflection = reflective_chain.invoke({
            "former_code": curr_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"]
        })

        curr_raw = generative_chain.invoke({"question": task["prompt"], "reference": reflection})
        curr_code = extract_code(curr_raw)

    if not solved:
        save_steps(task_id)  # mark as not solved


plot_step_changes("stats.json") #Showing plot















