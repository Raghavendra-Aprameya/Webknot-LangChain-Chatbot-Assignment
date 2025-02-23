
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# Set up the Gemini API key
os.environ["GOOGLE_API_KEY"] = "gemini_key"

# Load a sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load and preprocess the financial knowledge base
file_name = "financial_knowledge_base.txt"
with open(file_name, "r", encoding="utf-8") as file:
    financial_text = file.readlines()

financial_text = [line.strip() for line in financial_text if line.strip()]

# Convert text data to embeddings
text_embeddings = embedder.encode(financial_text)

# Create a FAISS index for searching relevant text
dim = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(text_embeddings))

# Initialize the AI model with Gemini API
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.5)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Relevant financial knowledge:\n{context}\n\nUser Question: {question}\n\nFinancial Insight:"
)

# API keys
STOCK_API_KEY = "1V9ONGVF1I79RQ9F"
NEWS_API_KEY = "6e83d9ac7c2b45c28e1306c9b63c22ad"

def fetch_stock_price(symbol: str) -> str:
    """Fetch stock price from Alpha Vantage API."""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("Global Quote", {})
        return f"The current price of {symbol} is ${data['05. price']}" if "05. price" in data else f"No price data found for {symbol}."
    except requests.RequestException as e:
        return f"Error fetching stock price: {str(e)}"

def fetch_stock_news(query: str) -> str:
    """Fetch news articles from News API."""
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])[:5]
        if not articles:
            return f"No news found for {query}."
        
        return "\n\n".join([f"{idx + 1}. {article['title']}\n   {article.get('description', 'No description available.')}" for idx, article in enumerate(articles)])
    except requests.RequestException as e:
        return f"Error fetching news: {str(e)}"

def retrieve_financial_info(query, top_k=3):
    """Retrieves the most relevant financial insights based on the user query."""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    retrieved_info = [financial_text[i] for i in indices[0]]

    context = "\n".join(retrieved_info)
    final_prompt = prompt_template.format(context=context, question=query)

    response = llm.predict(final_prompt)
    return response

def handle_general_financial_queries(query: str) -> str:
    return retrieve_financial_info(query)

class FinancialAgent:
    def __init__(self):
        self.tools = [
            Tool(name="Stock Price Checker", func=fetch_stock_price, description="Get the current stock price. Input should be a stock symbol (e.g., AAPL, GOOGL)."),
            Tool(name="Stock News Fetcher", func=fetch_stock_news, description="Get recent news about a stock or company. Input should be a company name or stock symbol."),
            Tool(name="General Financial Knowledge", func=handle_general_financial_queries, description="Retrieve general financial knowledge. Input should be a finance-related question."),
        ]

        # Use Gemini API model here
        self.llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.5)

        self.prompt = PromptTemplate.from_template("""
            You are a helpful financial assistant. Use the available tools to help with financial queries.
            Always provide clear, concise responses. Format amounts with appropriate currency symbols.

            Available tools: {tools}

            Use this format:
            Question: {input}
            Thought: think about what to do
            Action: choose an action from [{tool_names}]
            Action Input: the input for the action
            Observation: the result of the action
            ... (repeat Thought/Action/Action Input/Observation if needed)
            Thought: I now know the final answer
            Final Answer: the final answer to the original question

            Begin!

            Question: {input}
            {agent_scratchpad}
        """)

        self.agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True, max_iterations=3)

    def process_query(self, query: str) -> str:
        """Process a user query and return the response."""
        try:
            response = self.agent_executor.invoke({"input": query})
            return response.get("output", "No response generated.")
        except Exception as e:
            return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    agent = FinancialAgent()
    
    while True:
        user_input = input("\nAsk a financial question (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.process_query(user_input)
        print("\nAssistant:", response)
