# from langchain.tools.retriever import create_retriever_tool
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.output_parsers import StrOutputParser
# from api_key import *

# from langchain.tools import Tool
# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.load import dumps, loads

# from langchain_core.chat_history import (
#     BaseChatMessageHistory,
#     InMemoryChatMessageHistory,
# )
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langgraph.prebuilt import create_react_agent


# class GradeDocuments(BaseModel):
#     binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# class GradeHallucinations(BaseModel):
#     binary_score: str =  Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# class ChatModel:
#     def __init__(self):
#         self.store = {}
#         self.MAX_HISTORY_LENGTH = 10
#         self.init_vector_store()
#         self.init_rag_retriever_chains()
#         self.init_rag_retriever_tool()
#         self.init_react_agent()
    
#     def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
#         if session_id not in store:
#             self.store[session_id] = InMemoryChatMessageHistory()  # Each session has separate memory
#         return self.store[session_id][-self.MAX_HISTORY_LENGTH:]

#     def update_vector_store(self):
#         pass 
    
#     def init_vector_store(self):
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         vectorstore = Chroma(persist_directory="/home/guiseai/llm/agent_rag/chroma_db", 
#                             embedding_function=embeddings) # OpenAIEmbeddings(model="text-embedding-ada-002")
#         self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
#     def reciprocal_rank_fusion(self,results: list[list], k=60):
#         """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
#             and an optional parameter k used in the RRF formula """
        
#         fused_scores = {}
#         for docs in results:
#             for rank, doc in enumerate(docs):
#                 doc_str = dumps(doc)
#                 if doc_str not in fused_scores:
#                     fused_scores[doc_str] = 0
#                 previous_score = fused_scores[doc_str]
#                 fused_scores[doc_str] += 1 / (rank + k)

#         # Sort the documents based on their fused scores in descending order to get the final reranked results
#         reranked_results = [
#             (loads(doc), score)
#             for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#         ]

#         return reranked_results

#     def init_rag_retrieval_chains(self):
#         ################# MULTI QUERY and RE-RANKING / RAG FUSION#####################
#         multi_query_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
#                         Generate multiple search queries related to: {question} \n
#                         Output (4 queries):"""
#         multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)

#         llm = ChatOpenAI(model="gpt-4o-mini")

#         multi_query_llm = (
#             multi_query_prompt 
#             | llm
#             | StrOutputParser() 
#             | (lambda x: x.split("\n"))
#         )

#         self.retrieval_chain_rag_fusion = multi_query_llm | self.retriever.map() | self.reciprocal_rank_fusion
        
#         ##################Grade Document###########

#         filter_document_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeDocuments)

#         system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
#                     If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
#                     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

#         grade_prompt = ChatPromptTemplate.from_messages(
#                     [
#                         ("system", system),
#                         ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#                     ]
#                 )

#         self.filter_document_chain = grade_prompt | filter_document_llm

#     def data_retriever(query: str):
#         '''query expansion/transformation and rag-fusion'''
#         results = self.retrieval_chain_rag_fusion.invoke({"question":query})
        
#         '''Filter documents - Keep only relevant documents'''
#         docs = []
#         for res,score in results:
#             grade = self.filter_document_chain.invoke({"document":res.page_content ,"question":query})
#             if grade.binary_score=="yes":
#                 docs.append(res.page_content)

#         return docs[:5] 

#     def init_rag_retriever_tool(self):
#         retriever_tool = Tool.from_function(
#             func=self.data_retriever,  # The function that the tool will use
#             name="data_retriever",  # Name of the tool
#             description="Searches and returns excerpts from the PDFs"  # Description
#         )
#         self.tools = [retriever_tool]
 
#     def init_react_agent(self):
#         llm = ChatOpenAI(model="gpt-4o-mini")
#         self.agent_executor = create_react_agent(llm, tools) 
    
#     def init_hallucination_chain(self):
#         system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
#                         Give a binary score 'yes' or 'no'. 
#                         'Yes' means that the answer is grounded in / supported by the set of facts."""
#         hallucination_prompt = ChatPromptTemplate.from_messages(
#                     [
#                         ("system", system),
#                         ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
#                     ]
#                 )

#         llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

#         self.hallucination_grader_chain = hallucination_prompt | llm_hallucination_grader

#     def init_trimmer(self):
#         self.trimmer = trim_messages(
#                 messages,
#                 max_tokens=56,
#                 strategy="last",
#                 token_counter=ChatOpenAI(model="gpt-4o-mini"),
#                 include_system=False,
#                 allow_partial=True,
#             )

#     def chatbot(self,session_id,query):
#         messages = self.get_session_history(session_id)
#         messages.add_user_message(query)
        
#         messages = trim_messages(
#                 messages,
#                 max_tokens=56,
#                 start_on="human",
#                 end_on=("human", "tool"),
#                 strategy="last",
#                 token_counter=ChatOpenAI(model="gpt-4o-mini"),
#                 include_system=False,
#                 allow_partial=True,
#             )

#         ans = self.agent_executor.invoke({"messages":messages.messages})
#         messages.add_ai_message(ans["messages"][-1])
#         return ans["messages"][-1]


