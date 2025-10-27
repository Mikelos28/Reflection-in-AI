from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



model = OllamaLLM(model="gemma3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(collection_name="llm_memory", embedding_function=embeddings, persist_directory="./chroma_db")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
)

template1 = """
You are a master of coding and can help anyone find errors in their code.
Your job is to fix the given code, explain it with comments, and also
generate code based on the user's request.

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
Step 2: Update the code based on your review and output the improved version only.

Here is the code you generated before:
{former_code}  
"""


generative_prompt = ChatPromptTemplate.from_template(template1)
generative_chain = generative_prompt | model
reflective_prompt = ChatPromptTemplate.from_template(template2)
reflective_chain = reflective_prompt | model


def add_to_memory(question, answer):
    # Combine question and answer
    full_text = f"Question: {question}\nAnswer: {answer}"

    # Split into chunks
    chunks = text_splitter.split_text(full_text)

    # Store chunks in the vectorstore
    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"question": question}] * len(chunks),
    )


def get_relevant_context(question, k=5):
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([r.page_content for r in results])
    return context

# Main loop
while True:
    print("\n\n")
    question = input("I can help you with your code. To quit press q.\n> ") #Quit Option
    if question.lower() == "q":
        break

    reference = get_relevant_context(question)
    resultg = generative_chain.invoke({"question": question, "reference": reference})
    resulti = resultg

    for i in range(10):
        resultr = reflective_chain.invoke({"former_code": resultg})
        resulti = resultr

    print("\n---\nGenerated Answer:\n")
    print(resultr)
    print("\n---")

    add_to_memory(question, resultr)

