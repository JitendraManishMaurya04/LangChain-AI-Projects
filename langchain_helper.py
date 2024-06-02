# from langchain.llms.openai import OpenAI
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os
# from secret_key import openapi_key
from dotenv import load_dotenv

#Loads environment variables from .env file
load_dotenv()
# os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.7)

def generate_restrauntName_and_menuItems(cuisine):
    #Chain-1: Restraunt Name
    promptTemplateName = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restraunt for {cuisine} food. Suggest a fancy name for this."
    )
    nameChain = LLMChain(llm=llm, prompt=promptTemplateName, output_key="restrauntName")

    # Chain-2: Menu Items
    promptTemplateItems = PromptTemplate(
        input_variables=['restrauntName'],
        template="Suggest me some food menu items for {restrauntName} food. Return it as comma separated."
    )
    food_Items_Chain = LLMChain(llm=llm, prompt=promptTemplateItems, output_key="menuItems")

    chain = SequentialChain(chains=[nameChain, food_Items_Chain], input_variables=['cuisine'],output_variables=['restrauntName', 'menuItems'])
    openAIResponse =  chain({'cuisine': 'Indian'})
    return openAIResponse

if __name__ == "__main__":
    print(generate_restrauntName_and_menuItems("Italian"))