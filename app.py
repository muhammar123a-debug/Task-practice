import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt template summarization
summary_prompt = PromptTemplate(
    input_variable = ["text"],
    template="summarize this text in 2 lines:\n\n{text}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Chain template title
title_prompt = PromptTemplate(
    input_variable = ["text"],
    template="Write a short catchy title:\n\n{text}"
)
title_chain = LLMChain(llm=llm, prompt=title_prompt)

# memory 
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# --- Demo Run ---
print("---- Prompt Example ----")
print(summary_chain.run("Artificial Intelligence is shaping the future."))

print("\n---- Chain Example ----")
text = "LangChain helps developers build LLM-powered apps using chains and memory."
print(title_chain.run(text))

print("\n---- Memory Example ----")
print(conversation.run("My name is Ammar."))
print(conversation.run("I am learning LangChain."))
print(conversation.run("What is my name?"))