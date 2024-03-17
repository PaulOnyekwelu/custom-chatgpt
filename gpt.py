from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


load_dotenv(find_dotenv(), override=True)

memory_key = "chat_history"


llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=1)

history = FileChatMessageHistory("chat_history.json")

memory = ConversationBufferMemory(
    memory_key=memory_key, chat_memory=history, return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessage(content="You are chatbot having a conversation with a Human."),
        MessagesPlaceholder(variable_name=memory_key),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)


chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

while True:
    content = input("Your message:")
    if content in ["quit", "close", "bye", "end"]:
        print("Good bye!")
        break
    response = chain.invoke({"content": content})
    print(f"Chatbot: {response['text']}", flush=True)
    print("-" * 50)
