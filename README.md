# LangChain-AI-Projects

## Contains Below LangChain Projects:
### 1) Restaurant Name and Menu Items Generator
    It uses OpenAI LLM for getting answer for the prompt provided.
    Uses StreamLit for creating quick POC UI application.

### 2) Equity Research Tool
    It uses OpenAI LLM for getting answer for the prompt provided.
    FAISS(Facebook AI Similarity Search) is used for Vector Embedding and also as a in-memory Vector DB.
    Research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.
#### Features
        Load URLs or upload text files containing URLs to fetch article content.
        Process article content through LangChain's UnstructuredURL Loader
        Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
        Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.
	
### 3) Retailer Q&A Tool
    This is an end to end LLM project based on Google PaLM and Langchain. 
    We are building a system that can talk to MySQL database. 
    User asks questions in a natural language and the system generates answers by converting those questions to an SQL query and then executing that query on MySQL database.
    4Tees is a T-shirt store where they maintain their inventory, sales and discounts data in MySQL database. A store manager will may ask questions such as,
        -> How many white color Adidas t shirts do we have left in the stock?
        -> How much sales our store will generate if we can sell all extra-small size t shirts after applying discounts? The system is intelligent enough to generate accurate queries for given question and execute them on MySQL database
#### Project Highlights
    -> We will build an LLM based question and answer system that will use following,
    -> Google Palm LLM
    -> Hugging face embeddings
    -> Streamlit for UI
    -> Langchain framework
    -> Chromadb as a vector store
    -> Few shot learning
