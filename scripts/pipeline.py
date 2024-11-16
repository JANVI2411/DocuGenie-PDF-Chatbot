from langgraph_rag import *
import json 
from pymongo import MongoClient

class MongoDBHandler:
    def __init__(self, uri="mongodb://localhost:27017", db_name="pdf_chatbot"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def save_session(self, session_id, chat_history):
        """Save session state to MongoDB."""
        # chat_history = json.dumps(chat_history)
        self.db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"chat_history": chat_history}},
            upsert=True
        )

    def load_session(self, session_id):
        """Load session state from MongoDB."""
        session = self.db.sessions.find_one({"session_id": session_id})
        return session["chat_history"] if session else []
        # return json.loads(session["chat_history"]) if session else []

    def delete_session(self, session_id):
        """Delete a session from MongoDB."""
        self.db.sessions.delete_one({"session_id": session_id})


class State(TypedDict):
    # Sequence[BaseMessage]
    questions: Annotated[list[AnyMessage], add_messages]
    docs: Annotated[list[AnyMessage], add_messages]
    filtered_docs: Annotated[list[AnyMessage], add_messages]
    generation: Annotated[list[AnyMessage], add_messages]
    chat_history: list[str]

class ChatBotModelGraph(ChatBotModel):
    def __init__(self):
        super().__init__()
        self.workflow = StateGraph(State)
        self.mongodb = MongoDBHandler()
        self.filter_loop_count = 0
        self.hallucination_loop_count = 0
        self.build_graph()

    def contextualize_question(self,state):
        print("---Contextualize Question---")
        query = state["questions"][-1].content  # Using questions instead of messages
        chat_history = state["chat_history"]
        response = self.contextualize_q_chain.invoke({"chat_history": chat_history, "question": query})
        state["questions"] = [AIMessage(content=response)]
        return state

    def document_retriever(self, state):
        print("---Docs Retriever---")
        
        query = state["questions"][-1].content  # Using questions instead of messages
        response = self.retriever.get_relevant_documents(query)
        
        docs = [res.page_content for res in response]
        state["docs"] = docs  # Store retrieved documents in `docs`
        
        return state

    def filter_documents(self, state):
        print("---FILTER DOCUMENTS---")
        
        query = state["questions"][-1].content
        docs = state["docs"]
        filtered_docs = []
        
        for doc in docs:
            res = self.filter_document_chain.invoke({"document": doc, "question": query})
            score = res.binary_score
            print("########## score:", score)
            if score == "yes":
                filtered_docs.append(doc)
        
        state["filtered_docs"] = filtered_docs  # Store filtered documents in `filtered_docs`
        return state

    def filter_condition_node(self, state):
        print("---FILTER CONDITION NODE---")
        
        if state["filtered_docs"]:
            self.filter_loop_count = 0
            return "rag_generation"
        
        if self.filter_loop_count >= 1:
            self.filter_loop_count = 0
            state["generation"] = [AIMessage(content="No relevant data found from the PDF.")]
            print("1111111111",state)
            return END
        else:
            self.filter_loop_count += 1
            print("222222222",state)
            return "transform_query"

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        
        question = state["questions"][-1].content
        new_question = self.rewrite_question_chain.invoke({"question": question})
        print("------NEW QUERY:", new_question)
        
        state["questions"].append(AIMessage(content=new_question))  # Add transformed query to questions
        return state

    def rag_generation(self, state):
        print("---RAG GENERATION---")
        # print("State:\n",state["filtered_docs"])
        # print("\n######################\n")
        question = state["questions"][-1].content
        docs = "\n".join(state["filtered_docs"][-1].content)
        # chat_history = state["chat_history"]
        generation = self.rag_chain.invoke({"context": docs, "question": question})
        print("########", generation)
        
        state["generation"] = [generation]  # Store generated response
        return state

    def hallucination_grader(self, state):
        print("---HALLUCINATION GRADER---")
        
        docs = "\n".join(state["filtered_docs"][-1].content)
        generation = state["generation"][-1].content
        grade = self.hallucination_grader_chain.invoke({"documents": docs, "generation": generation})
        
        if grade.binary_score == "yes":
            self.hallucination_loop_count = 0
            return END
        
        if self.hallucination_loop_count == 1:
            self.hallucination_loop_count = 0
            state["generation"] = [AIMessage(content="[Hallucinated]")]
            return END
        
        self.hallucination_loop_count += 1
        return "rag_generation"

    def build_graph(self):
        self.workflow.add_node("contextualize_question", self.contextualize_question)
        self.workflow.add_node("document_retriever", self.document_retriever)
        self.workflow.add_node("filter_documents", self.filter_documents)
        self.workflow.add_node("transform_query", self.transform_query)
        self.workflow.add_node("rag_generation", self.rag_generation)
        self.workflow.add_node("hallucination_grader", self.hallucination_grader)

        self.workflow.add_edge(START, "contextualize_question")
        # self.workflow.add_edge("contextualize_question",END)

        self.workflow.add_edge("contextualize_question","document_retriever")
        self.workflow.add_edge("document_retriever", "filter_documents")
        self.workflow.add_conditional_edges("filter_documents", self.filter_condition_node)
        self.workflow.add_edge("transform_query", "document_retriever")
        self.workflow.add_conditional_edges("rag_generation", self.hallucination_grader)
        
        self.app = self.workflow.compile()

class ManageHistory:
    def __init__(self):
        self.MAX_CHAT_HISTORY_LENGTH = 4
        self.thresh = 2
        openai_model,temperature = 'gpt-4o-mini', 0
        self.llm = ChatOpenAI(model=openai_model, temperature=temperature)
        
        self.system_prompt = """Given the chats between human and AI, your task is to summarize 
                        their conversation into 50 tokens """
        self.prompt = ChatPromptTemplate.from_messages(
                        [("system",self.system_prompt ),
                        ("user","Chat: {chat}")])
        self.summary_llm = self.prompt | self.llm | StrOutputParser()

    def summarize_chat_history(self,chat):
        summary = self.summary_llm.invoke({"chat":chat})
        return summary

    def manage_chat_history(self,chat_history):
        if len(chat_history) >= self.MAX_CHAT_HISTORY_LENGTH:
            chat_history_summary = self.summarize_chat_history(chat_history[:self.thresh])
            chat_history = [f"Chat Summary: {chat_history_summary}"] + chat_history[self.thresh:]
        return chat_history

if __name__ == "__main__":
    chatbot = ChatBotModelGraph()
    manage_history = ManageHistory()
    # chatbot.mongodb.delete_session("user_1")
    # chatbot.mongodb.delete_session("user_2")
    query = "what is your name?"
    chat_history = chatbot.mongodb.load_session(session_id="user_2")
    
    chat_history = manage_history.manage_chat_history(chat_history)
    
    print(chat_history)
    # question = AIMessage(content=query)
    # response = chatbot.app.invoke({"chat_history":chat_history,"questions": [question]})
    
    # print("##################")
    # print(response)
    # print("##################\n")

    # chat_history.append(f"Human: {query}")
    # chat_history.append(f"Answer: {response['generation'][-1].content}")
    # # print("chat_history",chat_history)
    # chatbot.mongodb.save_session(session_id="user_2", chat_history = chat_history)
    
#memory management: Cut off the message + summarize the msg