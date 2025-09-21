import os 
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv("GOOGLE_API_KEY") )

summary_prompt = PromptTemplate(
    input_variables = ["text"],
    template = "Summarize text this 2 lines: \n\n{text}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

print("Summary:" , summary_chain.run("Artificial Intelligence is changing industries..."))

translate_prompt = PromptTemplate(
    input_Variable = ["text"],
    template ="Translate this text English into Urdu: \n\n{text}"
)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)
print("Urdu:", translate_chain.run("Knowledge is power."))

#question generator
question_prompt = PromptTemplate(
    input_variable = ["topic"],
    template="Generate 3 question about {topic}"
)
question_chain = LLMChain(llm=llm, prompt=question_prompt)
print("Question\n",question_chain.run("LangChain"))