from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0)
#  New, secure line:
# prompt = hub.pull("rlm/rag-prompt", dangerously_pull_public_prompt=True)

# Replaces the crashing hub.pull line completely
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
    ("human", "{question}")
])

generation_chain = prompt | llm | StrOutputParser()
