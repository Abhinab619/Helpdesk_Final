from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

app = FastAPI()

# Allow frontend requests (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model to receive messages from frontend
class Message(BaseModel):
    text: str

# Absolute paths for embedding storage
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")

# Tool 1.2
vectorstore2 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool2"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever2 = vectorstore2.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool2 = create_retriever_tool(retriever=retriever2,                           
                                       name="Udyami_Yojna_scheme1_SC_ST",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.3
vectorstore3 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool3"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever3 = vectorstore3.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool3 = create_retriever_tool(retriever=retriever3,                           
                                       name="Udyami_Yojna_scheme2_Extremely_Backward_Class",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.4
vectorstore4 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool4"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever4 = vectorstore4.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool4 = create_retriever_tool(retriever=retriever4,                           
                                       name="Udyami_Yojna_scheme3_YUVA",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.5
vectorstore5 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool5"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever5 = vectorstore5.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool5 = create_retriever_tool(retriever=retriever5,                           
                                       name="Udyami_Yojna_scheme4_Mahila",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.6
vectorstore6 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool6"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever6 = vectorstore6.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool6 = create_retriever_tool(retriever=retriever6,                           
                                       name="Udyami_Yojna_scheme5_Alpsankhyak",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Direct Gemini Tool
chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                              google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE")

@tool
def direct_llm_answer(query: str) -> str:
    """Directly generates an answer from the LLM."""
    response = chat.invoke(query)
    return response

# Adding a memory for context retention
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=chat, memory_key="chat_history", return_messages=True)

tools = [retriever_tool2, retriever_tool3, retriever_tool4, retriever_tool5, retriever_tool6, direct_llm_answer]

chat_prompt_template = hub.pull("hwchase17/openai-tools-agent")

# Create tool-calling agent
agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=chat_prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, memory=memory)

interaction_count = 0
MAX_MEMORY_SIZE = 5  # Clear memory after 5 interactions

@app.post("/chat")
def chat_with_model(msg: Message):
    global interaction_count, memory, agent_executor  # Ensure we can modify memory
    
    # Reset memory if it gets too large
    if interaction_count >= MAX_MEMORY_SIZE:
        print('Memory Reset')
        memory.clear()  # Clears stored history
        memory = ConversationSummaryMemory(llm=chat, memory_key="chat_history", return_messages=True)  # Reinitialize memory
        agent_executor = AgentExecutor(                                         # Reinitialized agent executor
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            memory=memory
        )
        interaction_count = 0  # Reset counter

    response = agent_executor.invoke({"input": msg.text})
    interaction_count += 1  # Increase interaction count
    
    return {
        "response": response.get("output", "No response generated"),
        "intermediate_steps": response.get("intermediate_steps", [])
    }
