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

STATS_FILE1 = "stats.json" #JSON FILE
STATS_FILE2 = "LongTerm.json" #JSON FILE FOR LONG TERM MEMORY RESULTS

#Resets json files so that they can keep track of the new data from the beginning
def reset_stats_files():
    with open(STATS_FILE1, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)


reset_stats_files()

# Create stats.json if missing
if not os.path.exists(STATS_FILE1):
    with open(STATS_FILE1, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)



# JSON Logging
def save_steps(problem_id, initial=None, reflection=None, loop=None, long_term=None, reflection_with_solution=None, solved_step=None):
    """Append or update a record in stats.json for this problem."""
    with open(STATS_FILE1, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False

    for entry in data:
        if entry["problem_id"] == problem_id:
            if initial is not None:
                entry["initial_solution"] = initial.strip()
            if reflection is not None:
                entry["reflection"] = reflection.strip()
            if loop is not None:
                entry["loop_solution"] = loop.strip()
            if long_term is not None:
                entry["long_term"] = long_term.strip()
            if reflection_with_solution is not None:
                entry["reflection_with_solution"] = reflection_with_solution.strip()
            if solved_step is not None:
                entry["solved_step"] = solved_step

            updated = True
            break

    if not updated:
        data.append({
            "problem_id": problem_id,
            "initial_solution": initial or "",
            "reflection": reflection or "",
            "loop_solution": loop or "",
            "long_term": long_term or "",
            "reflection_with_solution": reflection_with_solution or "",
            "solved_step": solved_step
        })

    with open(STATS_FILE1, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

#PLOT FUNCTION
def plot_step_changes(json_path="stats.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

# ALL PLOT "COLUMNS"
    solved_without_reflection = 0
    solved_after_one_reflection = 0
    solved_after_loop = 0
    solved_with_long_term = 0
    after_reflecting_on_solution = 0
    not_solved = 0

    for entry in data:
        step = entry.get("solved_step", None)

        if step == 1:
            solved_without_reflection += 1
        elif step == 2:
            solved_after_one_reflection += 1
        elif step == 3:
            solved_after_loop += 1
        elif step == 4:
            solved_with_long_term += 1
        elif step == 5:
            after_reflecting_on_solution += 1
        else:
            not_solved += 1

    labels = [
        "Solved w/o reflection",
        "Solved w/ one reflection",
        "Solved in reflection loop",
        "Solved with long term",
        "Solved after reflecting solution",
        "Not solved"
    ]
    values = [
        solved_without_reflection,
        solved_after_one_reflection,
        solved_after_loop,
        solved_with_long_term,
        after_reflecting_on_solution,
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
    print("Solved with long term:          ", solved_with_long_term)
    print("Solved after reflecting solution:   ", after_reflecting_on_solution)
    print("Not solved:                     ", not_solved)


dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test") #Used dataset

model = OllamaLLM(model="gemma3:latest") #Model used
embeddings = OllamaEmbeddings(model="mxbai-embed-large") #Embeddings used
vectorstore = Chroma(collection_name="llm_memory", embedding_function=embeddings, persist_directory="./chroma_db") #Vector Database

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

def add_to_memory(question, answer, taskid):
    full_text = f"Question: {question}\nAnswer: {answer}, taskid: {taskid}"
    chunks = text_splitter.split_text(full_text)
    ids = [f"{taskid}_{i}" for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=[{"question": question}] * len(chunks), ids=ids)

def get_relevant_context(question, k=10):
    results = vectorstore.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])

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

Step 2: Write down your suggestions for improvement in natural language.

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

template_4 = """
Imagine you are a programming expert. You will be given some code that solves a problem you will also be given. Your job is to 
reflect on that solution and think of reasons why this solution works and explain your thinking behind it.

Problem:
{problem}

Code:
{code}
"""
generative_prompt = ChatPromptTemplate.from_template(template1)
generative_chain = generative_prompt | model
reflective_prompt = ChatPromptTemplate.from_template(template2)
reflective_chain = reflective_prompt | model
no_reflection_prompt = ChatPromptTemplate.from_template(template3)
no_reflection_chain = no_reflection_prompt | model
correct_prompt = ChatPromptTemplate.from_template(template_4)
correct_chain = correct_prompt | model


# Extract code from the LLM answers
def extract_code(text):
    code = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if code: return code[0].strip()

    code = re.findall(r"```(.*?)```", text, re.DOTALL)
    if code: return code[0].strip()

    return text.strip()

# Testing the llm answers
def run_tests(solution_code, test_list, timeout_seconds=4):
    test_code = "\n".join(test_list)
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
total_tasks = 0

for task in dataset:

    task_id = task["task_id"]
    question = task["prompt"]
    test_code = task["test_list"]

    print(f"\nSolving {task_id} — No reflection - Problem no. {total_tasks}") #Current problem

    llm_raw = no_reflection_chain.invoke({"question": question}) #Llm's initial response
    init_code = extract_code(llm_raw)

    test_result = run_tests(init_code, test_code) #Running the test

    if test_result["passed"]:
        save_steps(task_id, initial=llm_raw, solved_step=1)
        add_to_memory(question, init_code, task_id) #Adding the correct answer and the question for use in the long term memory section
    else:
        false_tasks.append(task_id)
        memo[task_id] = {       #short term memory items for the next step
            "answer": llm_raw,
            "solution_code": init_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"]
        }
    total_tasks += 1

# ONE REFLECTION (step 2)
memo_reflection = {}
total_tasks2 = 0

for task2 in dataset:

    task_id = task2["task_id"]
    if task_id not in false_tasks:
        continue

    print(f"\nSolving {task_id} — STEP 2 (Reflection) - Problem no. {total_tasks2}")

    entry = memo[task_id]
    prev_code = entry["solution_code"] #Entry the code generated for this problem in the previous step

    # reflection
    reflection = reflective_chain.invoke({
        "former_code": prev_code,
        "stdout": entry["stdout"],
        "stderr": entry["stderr"]
    })  #Reflection on the code generated in the previous step before answering in the current step


    # new solution
    question = task2["prompt"]
    improved_raw = generative_chain.invoke({"question": task2["prompt"], "reference": reflection}) #Answer using the reflection conclusions
    improved_code = extract_code(improved_raw) #Getting the code part from the llm's answer in this step

    test_result = run_tests(improved_code, task2["test_list"])

    if test_result["passed"]:
        save_steps(task_id, reflection=improved_raw, solved_step=2) #If the test results are passed then we store the problem's id and record it was solved in this step
        false_tasks.remove(task_id)
        add_to_memory(question, reflection, task_id)
    else:
        memo_reflection[task_id] = improved_raw # Check ! If the test isn't passed, then we store the llm's raw answer for use in the next step
    total_tasks2 += 1

# REFLECTION LOOP (step 3)
total_tasks3 = 0

for task3 in dataset:

    task_id = task3["task_id"] #if task_id among the 200 problems is not in false tasks, continue
    if task_id not in false_tasks:
        continue

    question = task3["prompt"]
    print(f"\nSolving {task_id} — STEP 3 (Looped Reflection) - Problem no. {total_tasks3}")

    curr_raw = memo_reflection.get(task_id) #Get the failed answer from the previous step from short term memory
    curr_code = extract_code(curr_raw) #Get the answer's code

    solved = False #Is the problem solved

    for z in range(4):
        test_result = run_tests(curr_code, task3["test_list"])

        if test_result["passed"]:
            save_steps(task_id, loop=curr_raw, solved_step=3) #If the answer passes the test, store that the problem was solved in reflection loop step
            false_tasks.remove(task_id)
            add_to_memory(question, curr_code, task_id)
            solved = True
            break

        reflection = reflective_chain.invoke({
            "former_code": curr_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"]
        })

        curr_raw = generative_chain.invoke({"question": task3["prompt"], "reference": reflection})
        curr_code = extract_code(curr_raw)

    total_tasks3 += 1

# Long term memory / No reflection (Step 4)
total_tasks4 = 0
for task4 in dataset:
    task_id = task4["task_id"]
    if task_id not in false_tasks:
        continue
    question = task4["prompt"]
    test_code = task4["test_list"]
    reference = get_relevant_context(question)
    print(f"\n\nSolving {task_id} - 2nd half of the problem set \ long term memory use - Step 4")
    long_term_answer = generative_chain.invoke({"question": question, "reference": reference})
    long_term_code = extract_code(long_term_answer)
    test_result = run_tests(long_term_code, test_code)
    if test_result["passed"]:
        save_steps(task_id, long_term=long_term_answer, solved_step=4)
        false_tasks.remove(task_id)

#Reflection on correct code (Step 5)
total_tasks5 = 0
for task5 in dataset:
    task_id = task5["task_id"]
    if task_id not in false_tasks:
        continue
    question = task5["prompt"]
    test_code = task5["test_list"]
    correct_code = task5["code"] #The code that solves the problem
    print(f"\n\nSolving {task_id} - Reflection on correct answer - Step 5")
    judgement = correct_chain.invoke({"problem" : question, "code" : correct_code})
    llm_an = generative_chain.invoke({"question": question, "reference": judgement})
    an_code = extract_code(llm_an)
    test_result = run_tests(an_code, test_code)
    if test_result["passed"]:
        save_steps(task_id, llm_an, solved_step=5)
        false_tasks.remove(task_id)
    else:
        save_steps(task_id)

plot_step_changes("stats.json") #Showing plot




















