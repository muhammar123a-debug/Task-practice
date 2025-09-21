import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# first step 
summary_prompt = PromptTemplate(
    input_variables = ["text"],
    template = "Summarize this in 2 lines: \n\n {text}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

#Second step
title_prompt = PromptTemplate(
    input_variables = ["text"],
    template="Write a catchy title for this summary: \n\n{text}"
)
title_chain = LLMChain(llm=llm, prompt=title_prompt)

overall_chains = SimpleSequentialChain(
    chains=[summary_chain, title_chain],
    verbose = True
)

input_text = """
Artificial Intelligence is transforming industries by automating tasks, 
improving efficiency, and enabling smarter decision-making across the world.
"""
result = overall_chains.run(input_text)
print(result)

