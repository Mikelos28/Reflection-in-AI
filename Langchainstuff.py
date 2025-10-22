from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


model = OllamaLLM(model="gemma3") #The model that is being used
embeddings = OllamaEmbeddings(model="mxbai-embed-large") # The Embeddings
vectorstore = Chroma(collection_name="llm_memory", embedding_function=embeddings, persist_directory="./chroma_db") # Name, embeddings and directory of the vector data base

# The instructions 
template = """
You are a master of coding and can help anyone find errors in their code.
Your job is to fix the given code, explain it with comments, and also
generate code based on the user's request.

User question:
{question}

Relevant past answers:
{reference}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# A function that is being used to add the answer-question pair in the vector database
def add_to_memory(question, answer):
    vectorstore.add_texts(
        texts=[answer],
        metadatas=[{"question": question}],
    )

# The get context function, where data inside the vector database are retrieved to be used as context from the llm to answer the user's question
# It returns the 5 most relevant results from past pairs for context
def get_relevant_context(question, k=5):
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([r.page_content for r in results])
    return context

# The main loop where the user inputs the question and the llm answers and stores the question-answer pair in the vector database
while True:
    print("\n\n")
    question = input("I can help you with your code. To quit press q.\n> ")
    if question.lower() == "q":
        break

    reference = get_relevant_context(question)
    result = chain.invoke({"question": question, "reference": reference})

    print("\n---\nGenerated Answer:\n")
    print(result)
    print("\n---")


    add_to_memory(question, result)
