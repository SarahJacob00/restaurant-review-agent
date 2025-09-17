from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
    You are an expert at answering questions about a pizza restaurants
    
    Here are some reviews {reviews}
    
    Here is the question you need to answer{query}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while(True):
    query = input("What would you like to know? Enter q if you'd like to quit")
    if query == 'q':
        break
    
    reviews = retriever.invoke(query)
    result = chain.invoke({"reviews": reviews, "query":query})
    print(result)
