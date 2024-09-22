import os 
import time 
import threading

from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.tools.retriever import create_retriever_tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.llms.llamafile import Llamafile
# from langgraph.prebuilt import create_react_agent
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

class ChatBotModel:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_id = 'Qwen/Qwen2-1.5B-Instruct'
        # self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
        self.persist_directory = os.path.join(self.project_dir,"chroma_db")
        self.emb_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.pdf_status = ""
        self.qa_mem_chain = None
        self.vectordb = None 
        self.current_pdf_path = None 
        self.current_vectordb_path = None
        self.llm_status = "None" 
        
        self.monitor_thread = threading.Thread(target=self.load_model)
        self.monitor_thread.daemon = True  # Allows the thread to exit when the main program exits
        self.monitor_thread.start()
    
        # self.emb_thread = threading.Thread(target=self.load_embedding_model)
        # self.emb_thread.daemon = True  # Allows the thread to exit when the main program exits
        # self.emb_thread.start()
        # self.load_model()
        self.store = {}
        self.conversational_rag_chain = None 
        self.retriever = None 
        
    def pdf_parser(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.all_splits = text_splitter.split_documents(docs)
        self.current_pdf_path = pdf_path
        self.pdf_status = "Loaded"

    def load_embedding_model(self):
        
        self.hf_emb = HuggingFaceEmbeddings(
            model_name=self.emb_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    def get_vectorstore(self, db_name):
        """# Loading vectorstore and embedding model"""
        # while not self.hf_emb:
        #     time.sleep(2)
        self.hf_emb = HuggingFaceEmbeddings(
            model_name=self.emb_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        self.current_vectordb_path = os.path.join(self.persist_directory,db_name)
        self.vectordb = Chroma(persist_directory=self.current_vectordb_path, embedding_function=self.hf_emb)

        if self.vectordb._collection.count()==0:
            uuids = [str(i) for i in range(len(self.all_splits))]
            self.vectordb.add_documents(documents=self.all_splits, ids=uuids)
        
        self.pdf_status = "Processed"
        self.retriever = self.vectordb.as_retriever()

    def load_model(self):
        """# Local LLM"""
        # go for a smaller model if you dont have the VRAM
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100
            # truncation=True
        )

        self.local_llm = HuggingFacePipeline(pipeline=pipe)
        self.llm_status = "Loaded"
    
    def create_rag_chain(self,pdf_name):
        # History aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        if not self.local_llm:
            self.load_model()

        if not self.retriever:
            self.get_vectorstore(pdf_name)

        history_aware_retriever = create_history_aware_retriever(
            self.local_llm, self.retriever, contextualize_q_prompt
        )

        # stuff document LLM chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        
        # trimmer = trim_messages(
        #     max_tokens=1000,
        #     strategy="last",
        #     token_counter=self.local_llm,
        #     include_system=True,
        # )
        question_answer_chain = create_stuff_documents_chain(self.local_llm, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Memory chain
        

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def parse_answer(self,text, question):
        question = f"Human: {question}"
        start = text.find(question)
        if start != -1:
            extracted_text = text[start + len(question):]
            return extracted_text.strip()
        return "I'm unable to answer this question."

    def invoke_llm(self,question,pdf_name):
        if not self.conversational_rag_chain:
            self.create_rag_chain(pdf_name)

        s = time.time()
        print("Invoke LLM... ")
        ans = self.conversational_rag_chain.invoke( {"input": question},
                                                config={"configurable": {"session_id": "abc123"}} 
                                                )

        print("\n\n--------------Answer-----------------------", time.time() - s)
        return self.parse_answer(ans["answer"], question)

    
    # def create_llm_client(self, pdf_name):
        
    #     if not self.vectordb:
    #         self.get_vectorstore(pdf_name)
        
    #     self.memory = ConversationBufferMemory(
    #         memory_key="chat_history",
    #         return_messages=True 
    #     )

    #     self.qa_mem_chain = ConversationalRetrievalChain.from_llm(
    #         self.local_llm,
    #         retriever=self.vectordb.as_retriever(),
    #         memory=self.memory
    #     )
    #     # search_kwargs={"k": 5}
    #     # combine_docs_chain_kwargs={"prompt": self.QA_CHAIN_PROMPT}

    # def parse(self,text):
    #     res = ""
    #     if "Answer:" in text:
    #         res = text.split("Answer:")[-1].strip()
    #     return res

    # def llm_client_chat(self,question,pdf_name):
    #     if self.qa_mem_chain is None:
    #         self.create_llm_client(pdf_name)
    #     result = self.qa_mem_chain.invoke({"question": question})
    #     # result = {"answer": "Hi this is me, AI"}
    #     return self.parse(result["answer"])

# print("-----------------")
# ans = ChatBotModel().invoke_llm("Who is Janvi?","/home/guiseai/llm/pdf_summarizer/chroma_db/Janvi_Chokshi_CV.pdf")
# print(ans)