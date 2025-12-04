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

dataset = load_dataset("openai/openai_humaneval", split="test")

model = OllamaLLM(model="gemma3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(collection_name="llm_memory", embedding_function=embeddings, persist_directory="./chroma_db")

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

# Reflection template now receives stdout/stderr and former_code to analyze
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

# simple solver prompt
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


# Helpers (extract_code, memory)

def extract_code(text):
    # Try fenced python blocks first
    fence = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if fence:
        return fence[0].strip()
    # Try generic fences
    fence = re.findall(r"```(.*?)```", text, re.DOTALL)
    if fence:
        return fence[0].strip()
    # Fallback: return whole text
    return text.strip()

def add_to_memory(question, answer, taskid):
    full_text = f"Question: {question}\nAnswer: {answer}, taskid: {taskid}"
    chunks = text_splitter.split_text(full_text)
    ids = [f"{taskid}_{i}" for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=[{"question": question}] * len(chunks), ids=ids)

def get_relevant_context(question, k=10):
    results = vectorstore.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])

# run_tests: return stdout/stderr and result

def run_tests(solution_code, test_code, timeout_seconds=5):
    # Create temp file and write solution + tests
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
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
            text=True,  # ensures stdout/stderr are strings
        )
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": "",
            "stderr": f"TimeoutExpired: Code execution exceeded {timeout_seconds} seconds.",
            "returncode": None,
        }

    finally:
        # cleanup temp file if it still exists
        try:
            if os.path.exists(fname):
                os.remove(fname)
        except Exception:
            pass


# Main evaluation loops (refactor)
total_tasks = 0
correct_solutions = 0
false_tasks = []           
memo = {}                  # map task_id -> {"answer": <llm output>, "stdout":..., "stderr":...}

# Step 1: initial generation without reflection
for task in dataset:

    task_id = task["task_id"]
    question = task["prompt"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    print(f"\n Currently solving: {task_id} ({entry_point})")
    llm_response = no_reflection_chain.invoke({"question": question})
    solution_code = extract_code(llm_response)

    test_result = run_tests(solution_code, test_code)
    if test_result["passed"]:
        correct_solutions += 1
    else:
        false_tasks.append(task_id)
        memo[task_id] = {
            "answer": llm_response,
            "solution_code": solution_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"],
        }
    total_tasks += 1

success_rate = correct_solutions / total_tasks * 100 if total_tasks else 0.0
print("\n-----------")
print(f"\n\nThe LLM's success rate without using reflection and context is {success_rate:.2f}% ({correct_solutions}/{total_tasks})\n\n")
print("Any unsolved tasks from this step, will be taken to the one time reflection agent")
print("\n\n---------------\n\n")

# Step 2: one-time reflection
total_tasks2 = 0
correct_solutions2 = 0
memo_1_reflection = {}  # map task_id -> latest failed llm response

for task in dataset:
    if not false_tasks:
        break

    task_id = task["task_id"]
    if task_id not in false_tasks:
        continue

    question = task["prompt"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    print(f"\n Currently solving (one-time reflection): {task_id} ({entry_point})")

    # previous answer + logs
    prev = memo[task_id]
    prev_answer = prev["answer"]
    prev_code = prev["solution_code"]
    prev_stdout = prev["stdout"]
    prev_stderr = prev["stderr"]

    # Ask generative chain with reference to previous answer
    gen_resp = generative_chain.invoke({"question": question, "reference": prev_answer})
    init_code = extract_code(gen_resp)

    # Run tests on init_code to obtain real logs
    init_result = run_tests(init_code, test_code)
    if init_result["passed"]:
        correct_solutions2 += 1
        # mark solved: remove from false_tasks and update memo
        false_tasks.remove(task_id)
        memo[task_id] = {"answer": gen_resp, "solution_code": init_code, "stdout": init_result["stdout"], "stderr": init_result["stderr"]}
        total_tasks2 += 1
        continue

    # Give logs to the reflective agent
    reflection = reflective_chain.invoke({
        "former_code": init_code,
        "stdout": init_result["stdout"],
        "stderr": init_result["stderr"],
    })

    # Use reflection as reference to generate a corrected solution
    gen_after_reflect = generative_chain.invoke({"question": question, "reference": reflection})
    corrected_code = extract_code(gen_after_reflect)
    corrected_result = run_tests(corrected_code, test_code)

    if corrected_result["passed"]:
        correct_solutions2 += 1
        if task_id in false_tasks:
            false_tasks.remove(task_id)
        memo[task_id] = {"answer": gen_after_reflect, "solution_code": corrected_code, "stdout": corrected_result["stdout"], "stderr": corrected_result["stderr"]}
    else:
        # still failing, keep for the reflection loop step
        memo_1_reflection[task_id] = {
            "answer": gen_after_reflect,
            "solution_code": corrected_code,
            "stdout": corrected_result["stdout"],
            "stderr": corrected_result["stderr"],
        }

    total_tasks2 += 1

success_rate2 = (correct_solutions2 / total_tasks2 * 100) if total_tasks2 else 0.0
print("\n\n--------------\n\n")
print(f"The one time reflection agent manages to solve {success_rate2:.2f}% ({correct_solutions2}/{total_tasks2}) of the problems not solved in the previous step\n\n")
print("What problems remain unsolved will be solved by the reflection loop agent")
print("\n\n--------------\n\n")

# Step 3: iterative reflection loop (multiple iterations per problem)
total_tasks3 = 0
correct_solutions3 = 0
no_of_loops = 0

for task in dataset:
    if not false_tasks:
        break

    task_id = task["task_id"]
    if task_id not in false_tasks:
        continue

    question = task["prompt"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    print(f"\n Currently solving (iterative reflection loop): {task_id} ({entry_point})")

    # Start from last known response (from memo_1_reflection if present else memo)
    if task_id in memo_1_reflection:
        current_entry = memo_1_reflection[task_id]
    else:
        current_entry = memo[task_id]

    current_ans = current_entry["answer"]
    current_code = current_entry["solution_code"]
    current_stdout = current_entry.get("stdout", "")
    current_stderr = current_entry.get("stderr", "")

    short_term = []  # list of reflection outputs (strings)

    solved = False
    for i in range(10):
        # run tests for current code (could be redundant on first iter but ensures latest logs)
        test_result = run_tests(current_code, test_code)
        if test_result["passed"]:
            correct_solutions3 += 1
            solved = True
            if task_id in false_tasks:
                false_tasks.remove(task_id)
            break

        # get reflection using latest code & logs
        reflection = reflective_chain.invoke({
            "former_code": current_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"],
        })
        no_of_loops += 1
        short_term.append(reflection)

        # Build a compact "reference" for the generative chain:
        # join last few reflections and include previous solution and logs
        reference_payload = "\n\n".join(short_term[-3:])  # last 3 reflections
        reference_payload += "\n\nPrevious stdout:\n" + test_result["stdout"]
        reference_payload += "\n\nPrevious stderr:\n" + test_result["stderr"]
        # Ask the generative model for a new attempt
        gen_resp = generative_chain.invoke({"question": question, "reference": reference_payload})
        current_code = extract_code(gen_resp)
        current_ans = gen_resp

        # Save progress in memo_1_reflection for potential future loops
        memo_1_reflection[task_id] = {
            "answer": current_ans,
            "solution_code": current_code,
            "stdout": test_result["stdout"],
            "stderr": test_result["stderr"],
        }

    if not solved:
        pass

    total_tasks3 += 1

success_rate3 = (correct_solutions3 / total_tasks3 * 100) if total_tasks3 else 0.0
print("\n\n--------------\n\n")
print(f"The final agent managed to solve {success_rate3:.2f}% ({correct_solutions3}/{total_tasks3}) of the problems attempted in the reflection loop\n\n")
print(f"Number of loops used by the agent in this step: {no_of_loops}\n\n")
print("\n\n--------------\n\n")















