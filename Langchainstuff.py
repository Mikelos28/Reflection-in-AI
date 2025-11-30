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


# Load the full HumanEval dataset

dataset = load_dataset("openai/openai_humaneval", split="test")

# Each row in dataset is a dict like:
# {
#   "task_id": "HumanEval/0",
#   "prompt": "...",               # problem statement and function signature
#   "canonical_solution": "...",   # correct reference code
#   "test": "...",                 # test code to evaluate solution
#   "entry_point": "function_name" # name of the function to call
# }


# Initialize LLM and vector memory

model = OllamaLLM(model="gemma3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(collection_name="llm_memory", embedding_function=embeddings, persist_directory="./chroma_db")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

# -------------------------------
# Prompt templates
# -------------------------------
template1 = """
You are a master in solving programming problems. You will be given a series of programming problems
to solve.

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
Step 2: Write down your suggestions for improvement.

Here is the code the LLM generated before:
{former_code}  
"""

#Template for problem solving without reflection
template3 = """  
You are a master in solving programming problems. You will be given a series of programming problems
to solve.

User question:
{question}
"""

generative_prompt = ChatPromptTemplate.from_template(template1)
generative_chain = generative_prompt | model
reflective_prompt = ChatPromptTemplate.from_template(template2)
reflective_chain = reflective_prompt | model
no_reflection_prompt = ChatPromptTemplate.from_template(template3)
no_reflection_chain = no_reflection_prompt | model



# -------------------------------
# Memory helper functions
# -------------------------------
def add_to_memory(question, answer, taskid):
    full_text = f"Question: {question}\nAnswer: {answer}, taskid: {taskid}"
    chunks = text_splitter.split_text(full_text)
    ids = [f"{taskid}_{i}" for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=[{"question": question}] * len(chunks), ids=ids)

def get_relevant_context(question, k=10):
    results = vectorstore.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])

def extract_code(text):
    # Try fenced blocks first
    fence = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if fence:
        return fence[0].strip()

    # Try generic fences
    fence = re.findall(r"```(.*?)```", text, re.DOTALL)
    if fence:
        return fence[0].strip()

    # Fallback: return whole text
    return text.strip()

def run_tests(solution_code, test_code):
    # Create a safe temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # Write user's solution + tests
        f.write(solution_code)
        f.write("\n\n")
        f.write(test_code)
        f.flush()

        try:
            result = subprocess.run(
                ["python3", f.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

# -------------------------------
# Iterate through HumanEval tasks
# -------------------------------

correct_solutions = 0
total_tasks = 0
no_of_loops = 0 #The total number of loops used to reach the result in the end

# Reflection Loop (case 1)
for task in dataset:

    if total_tasks > 20:  #Used to stop after a certain amount of problems for demonstration purposes
        break            #Remove it to run for all the problems

    question = task["prompt"]
    reference_solution = task["canonical_solution"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    #Short term memory
    short_term = []

    print(f"\n Currently solving: {task['task_id']} ({entry_point})") # The problem we are currently at

    reference = get_relevant_context(question) # Gets context (if it exists)
    resultg = generative_chain.invoke({"question": question, "reference": short_term}) #Llm's initial response

    # Loop
    for i in range(10):
        solution_code = extract_code(resultg)  # Get the code from the llm's response
        is_correct = run_tests(solution_code, test_code)  # Run the llm's response as a subprocess
        if is_correct:
            correct_solutions += 1
            break
        resultr = reflective_chain.invoke({"former_code": solution_code}) #Reflection analysis
        short_term.append({"context": resultr})
        resultg = generative_chain.invoke({"question": question, "reference": short_term}) #The generative agent produces a new answer
        no_of_loops += 1


    total_tasks += 1


#Calculation of success rate
success_rate = correct_solutions / total_tasks * 100

print("--------------") #The llm's success rate on the problems
print(f"\n\nThe LLM's success rate using reflection is {success_rate:.2f}% ({correct_solutions}/{total_tasks})\n\n")
print(f"The number of refletion loops to get these results was {no_of_loops}")
print("\n--------------\n\n\n")


total_tasks2 = 0
correct_solutions2 = 0

#Asking the llm to solve a problem without reflection/context (case 2)
for task in dataset:

    if total_tasks2 > 20:  #Used to stop after a certain amount of problems for demonstration purposes
        break            #Remove it to run for all the problems

    question = task["prompt"]
    reference_solution = task["canonical_solution"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    print(f"\n Currently solving: {task['task_id']} ({entry_point})")  # The problem we are currently at
    resultg = no_reflection_chain.invoke({"question": question})  # Llm's initial response
    solution_code = extract_code(resultg)  # Get the code from the llm's response
    is_correct = run_tests(solution_code, test_code)  # Run the llm's response as a subprocess
    if is_correct:
        correct_solutions2 += 1
    total_tasks2 += 1

success_rate2 = correct_solutions2 / total_tasks2 * 100
print("\n-----------")
print(f"\n\nThe LLM's success rate without using reflection and context is {success_rate2:.2f}% ({correct_solutions2}/{total_tasks2})\n\n")
print("\n\n---------------\n\n")

#One time reflection agent (case 3)

total_tasks3 = 0
correct_solutions3 = 0

for task in dataset:

    if total_tasks3 > 20:
        break

    question = task["prompt"]
    reference_solution = task["canonical_solution"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    print(f"\n Currently solving: {task['task_id']} ({entry_point})")
    resultg = no_reflection_chain.invoke({"question": question})
    initcode = extract_code(resultg)
    resultr = reflective_chain.invoke({"former_code": initcode})
    resultg = generative_chain.invoke({"question": question, "reference": resultr})

    solution_code = extract_code(resultg)
    is_correct = run_tests(solution_code, test_code)
    if is_correct:
        correct_solutions3 += 1
    total_tasks3 += 1

success_rate3 = correct_solutions3 / total_tasks3 * 100
print("\n\n--------------\n\n")
print(f"The result when reflecting once is {success_rate3:.2f}%({correct_solutions3}/{total_tasks3})")
print("\n\n--------------\n\n")










