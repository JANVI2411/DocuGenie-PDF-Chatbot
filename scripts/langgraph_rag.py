import os 
import time 
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
import logging 

import json 
from typing import Annotated
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages

logger = logging.getLogger("rag_logs")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("backend.log")  # Log to a file
handler.setLevel(logging.INFO)

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# os.environ['OPENAI_API_KEY'] = "Your OpenAI API Key"

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class ChatBotModel:
    def __init__(self):
        openai_model,temperature = 'gpt-4o-mini', 0
        self.model_id = 'gpt-4o-mini'
        
        # s = time.time()
        self.llm = ChatOpenAI(model=openai_model, temperature=temperature)
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.persist_directory = os.path.join(self.project_dir,"chroma_db")
        # self.vectorstore_path = "/home/guiseai/llm/openai/chrom_db_openai"
        self.vectorstore_path = "/home/guiseai/llm/agent_rag/chroma_db"
        # s = time.time()
        self.create_retriever_agent(self.vectorstore_path)

        self.init_structured_llm_ouput()
        self.init_prompt()
        self.init_chains()
        

    def init_structured_llm_ouput(self):
        #Structured LLM Output
        self.structured_llm_filter_document = self.llm.with_structured_output(GradeDocuments)
        self.structured_llm_answer_grader = self.llm.with_structured_output(GradeAnswer)
        self.structured_llm_hallucination_grader = self.llm.with_structured_output(GradeHallucinations)

    def init_prompt(self):
        # Prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        self.doc_filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
                Give a binary score 'yes' or 'no'. 
                Yes' means that the answer resolves the question."""

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
                Give a binary score 'yes' or 'no'. 
                'Yes' means that the answer is grounded in / supported by the set of facts."""
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        system_rewrite_question_prompt = """ You are a question re-writer that converts the input question to 
                                            a better version that is optimized for web-search. 
                                            Look at the input and try to reason about the underlying semantic intent meaning."""

        self.rewrite_question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_rewrite_question_prompt),
                ("human", "Initial Question: {question} \n\n Improved Question: "),
            ]
        )
        system_rag_msg = """You are an intelligent assistant. 
                            Use the provided context to answer the question. 
                            Keep your answers concise, relevant, and accurate."""
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_rag_msg),
                # MessagesPlaceholder("chat_history"),
                ("human","Context: {context} \n\n Question:{question} \n\n Answer:")
            ]
        )
        

    def init_chains(self):
        self.contextualize_q_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()
        self.filter_document_chain = self.doc_filter_prompt | self.structured_llm_filter_document
        self.answer_grader_chain = self.answer_prompt | self.structured_llm_answer_grader
        self.hallucination_grader_chain = self.hallucination_prompt | self.structured_llm_hallucination_grader
        self.rewrite_question_chain = self.rewrite_question_prompt | self.llm | StrOutputParser()
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
    
    def create_vector_store(self,user_id,pdf_id,pdf_path):
        #init the chromaDB here
        pass 
    
    def update_vector_store(self,user_id,pdf_id,pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(docs)
        
        # update the vectore store
        hf_emb = HuggingFaceEmbeddings(
            model_name=self.emb_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        vectordb = Chroma(persist_directory=self.vectordb_path, embedding_function=hf_emb)

        if True: # it should check whether this (user_id,pdf_id) is there or not
            uuids = [str(i) for i in range(len(all_splits))]
            vectordb.add_documents(documents=all_splits, ids=uuids)
        
        with Database() as db:
            db.update_record("pdf_status",{"user_id":user_id,"pdf_id":pdf_id},{"status":"Added to knwoledge base"}) 
     
    
    def create_retriever_agent(self,retriever_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = Chroma(persist_directory=retriever_path, 
                            embedding_function=self.embeddings) # OpenAIEmbeddings(model="text-embedding-ada-002")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})

        # tool = create_retriever_tool(
        #     retriever,
        #     "blog_post_retriever",
        #     "Searches and returns excerpts from the Autonomous Agents blog post.",
        # )
        # tools = [tool]
        # self.retriever_react_agent = create_react_agent(self.llm, tools)

# class State(TypedDict):
#     # Sequence[BaseMessage]
#     messages: Annotated[list[AnyMessage], add_messages]
#     question: Annotated[list[AnyMessage], add_messages]
#     docs: Annotated[list[AnyMessage], add_messages]
#     filtered_docs: Annotated[list[AnyMessage], add_messages]
#     generation: Annotated[list[AnyMessage], add_messages]

# class ChatBotModelGraph(ChatBotModel):
#     def __init__(self):
#         super().__init__() 
#         self.workflow = StateGraph(State)
#         self.filter_loop_count = 0
#         self.hallucination_loop_count = 0
#         self.build_graph()

#     def document_retriever(self,state):
#         print("---Docs Retriever---")
        
#         response = self.retriever.get_relevant_documents(state["messages"][0].content)
#         docs = []
#         for res in response:
#             docs.append(res.page_content)
#         docs = json.dumps(docs)
#         msg = AIMessage(docs)
#         return {"messages":[msg]}

#     def filter_documents(self,state):
#         print("---FILTER DOCUMENTS---")
        
#         query = state["messages"][0].content
#         docs = state["messages"][-1].content
#         docs = eval(docs)
#         filter_docs = []
#         for doc in docs:
#             res = self.filter_document_chain.invoke({"document":doc ,"question":query})
#             score = res.binary_score 
#             print("##########score:",score)
#             if score=="yes":
#                 filter_docs.append(doc) 
#         docs = "\n".join(filter_docs)
#         state["messages"][-1].content = docs 
#         return state

#     def filter_condition_node(self,state):
#         print("---filter_condition_node---")
        
#         docs = state["messages"][-1].content
#         if len(docs):
#             self.filter_loop_count=0
#             return "rag_generation"
        
#         if self.filter_loop_count>=3:
#             self.filter_loop_count = 0
#             state["messages"][-1].content = "No relevant data found from the PDF."
#             return END          
#         else:
#             self.filter_loop_count+=1
#             return "transform_query"

#     def transform_query(self,state):
#         print("---TRANSFORM QUERY---")
        
#         question = state["messages"][0].content

#         new_question = self.rewrite_question_chain.invoke({"question":question})
#         print("------NEW QUERY:",new_question)
#         state["messages"][0].content = new_question
#         return state
        
#     def rag_generation(self,state):
#         print("---RAG GENERATION---")
        
#         question = state["messages"][0].content
#         docs = state["messages"][-1].content
#         generation = self.rag_chain.invoke({"context":docs,"question":question})
#         print("########",generation)
#         return {"messages":[AIMessage(generation)]}

#     def hallucination_grader(self,state):
#         print("---hallucination_grader---")
        
#         docs = state["messages"][-2].content
#         generation = state["messages"][-1].content
#         grade = self.hallucination_grader_chain.invoke({"documents":docs,"generation":generation})
#         # print("---grade.binary_score: "+grade.binary_score+"---")
#         if grade.binary_score == "yes":
#             self.hallucination_loop_count = 0
#             return END
#         if self.hallucination_loop_count==3:
#             self.hallucination_loop_count = 0
#             state["messages"][-1].content = "[Hallucinated]"
#             return END
#         self.hallucination_loop_count+=1
#         return "rag_generation"

#     def build_graph(self):
        
#         self.workflow.add_node("document_retriever", self.document_retriever)
#         self.workflow.add_node("filter_documents",self.filter_documents)
#         self.workflow.add_node("transform_query",self.transform_query)
#         self.workflow.add_node("rag_generation",self.rag_generation)
#         self.workflow.add_node("hallucination_grader",self.hallucination_grader)

#         self.workflow.add_edge(START, "document_retriever")
#         self.workflow.add_edge("document_retriever", "filter_documents")
#         self.workflow.add_conditional_edges("filter_documents",self.filter_condition_node)
#         self.workflow.add_edge("transform_query","document_retriever")
#         self.workflow.add_conditional_edges("rag_generation",self.hallucination_grader)
        
#         self.app = self.workflow.compile()

if __name__ == "__main__":
    chatbot = ChatBotModelGraph()
    query = "who is memory agent?"
    response = chatbot.app.invoke({"messages": [("user", query)]})
    print("##################")
    print(response)
    print("##################\n")

    # ans = chatbot.retriever_react_agent.invoke({"messages":[HumanMessage(content = "Hi GPT")]})
    # print(ans)
    # hallucination_grader.invoke({"documents": docs, "generation": generation})
    # print("\n--------------------------------\n")
    # ans = chatbot.retriever_react_agent.invoke({"messages":[HumanMessage(content = "what is memory agent?")]})
    # print(ans)
    # chatbot.question_router.invoke({"question": "Who will the Bears draft first in the NFL draft?"})
    # answer_grader.invoke({"question": question, "generation": generation})