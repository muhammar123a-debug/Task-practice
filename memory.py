import os
# from langchain.prompts import promptTemplate
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

#memory
memory = ConversationBufferMemory()
conversition = ConversationChain(llm=llm, memory=memory, verbose=True) 

# print(conversition.run("My name is Ammar"))
print(conversition.run("What is my name?"))
