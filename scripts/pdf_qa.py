from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os 

class ChatBotModel:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_id = 'Qwen/Qwen2-1.5B-Instruct'
        self.persist_directory = os.path.join(self.project_dir,"chroma_db")
        self.emb_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.pdf_status = ""
        self.qa_mem_chain = None
        self.vectordb = None 
        self.current_pdf_path = None 
        self.current_vectordb_path = None
        self.load_model()
        
    def pdf_parser(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.all_splits = text_splitter.split_documents(docs)
        self.current_pdf_path = pdf_path
        self.pdf_status = "Loaded"

    def get_vectorstore(self, db_name):
        """# Loading vectorstore and embedding model"""
        
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

    def load_model(self):
        """# Local LLM"""
        # go for a smaller model if you dont have the VRAM
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300
            # truncation=True
        )

        self.local_llm = HuggingFacePipeline(pipeline=pipe)

    # """# PromptTemplate"""
    def define_template(self):

        template = """Use the following pieces of context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    Use three sentences maximum.
                    Keep the answer as concise as possible.
                    Always say "thanks for asking!" at the end of the answer.

                    {context}

                    Question: {question}

                    Answer:"""

        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    def create_llm_client(self, pdf_name):
        
        if not self.vectordb:
            self.get_vectorstore(pdf_name)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True 
        )

        self.qa_mem_chain = ConversationalRetrievalChain.from_llm(
            self.local_llm,
            retriever=self.vectordb.as_retriever(),
            memory=self.memory
        )
        # search_kwargs={"k": 5}
        # combine_docs_chain_kwargs={"prompt": self.QA_CHAIN_PROMPT}

    def parse(self,text):
        res = ""
        if "Answer:" in text:
            res = text.split("Answer:")[-1].strip()
        return res

    def llm_client_chat(self,question,pdf_name):
        if self.qa_mem_chain is None:
            self.create_llm_client(pdf_name)
        result = self.qa_mem_chain.invoke({"question": question})
        # result = {"answer": "Hi this is me, AI"}
        return self.parse(result["answer"])