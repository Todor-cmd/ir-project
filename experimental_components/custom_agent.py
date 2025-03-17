from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph
from langchain_core.messages import HumanMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_core.retrievers import BaseRetriever


class AgentState(TypedDict):
    """State used for the custom agent."""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    context: Annotated[str, "Retrieved context from vector store"]
    response: Annotated[str, "Final response from LLM"]
    


class CustomAgent:
    """This class is used to create a custom agent langgraph graph, that uses a retriever to retrieve 
    context from a vector store and then uses an LLM to generate a response. It does not maintain memory."""
    
    def __init__(self, llm : BaseLLM, retriever : BaseRetriever):
        """Initialise the custom agent with an LLM and a retriever."""
        self.llm = llm
        self.retriever = retriever

    def retrieve(self, state: AgentState) -> AgentState:
        """A node in the graph that retrieves the context from the vector store."""
        # Get the last message
        last_message = state["messages"][-1].content
        # Do retrieval
        docs = self.retriever.invoke(last_message)
        # Join the docs into context
        context = "\n\n".join([doc.page_content for doc in docs])
        # Add context to state
        state["context"] = context
        return state


    def generate(self, state: AgentState) -> AgentState:
        """A node in the graph that generates a response using the LLM."""
        # Construct the prompt
        messages = state["messages"]
        context = state["context"]
        print (context)
        
        system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question:
        
        Context: {context}
        
        Answer the question based on the context provided. If you cannot find the answer in the context, say so."""
        
        # Generate response using the LLM
        response = self.llm.invoke([
            HumanMessage(content=system_prompt),
            *messages
        ])
        
        return {"response": response.content, **state}
    
    def get_agent(self):
        """Create the custom agent graph and return it.
        
        This method constructs a directed graph workflow for the custom agent.
        The graph consists of two nodes - retrieve and generate, connected sequentially.
        The retrieve node gets context from the vector store, while the generate node 
        produces the response using the LLM.
        
        Returns:
            Graph: A compiled langgraph Graph object that can be invoked to process messages.
        """
        # Create the graph
        workflow = Graph()

        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        # Add edges
        workflow.add_edge("retrieve", "generate")

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Set exit point
        workflow.set_finish_point("generate")

        # Compile the graph
        agent = workflow.compile()
        return agent