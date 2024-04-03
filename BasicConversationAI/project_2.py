from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory,ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

# memory = ConversationBufferMemory(
#     chat_memory = FileChatMessageHistory("messages.json"),
#     memory_key = "messages", 
#     return_messages = True
#     )

memory = ConversationSummaryMemory(
    llm = chat,
    memory_key = "messages", 
    return_messages = True
)
 
prompt = ChatPromptTemplate(
    input_variables = ["content","messages"],
    messages = [
        MessagesPlaceholder(variable_name = "messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    memory = memory,
    verbose = True
)

while True:
    content = input(">> ")
    result  = chain({"content": content})
    print(result['text'])