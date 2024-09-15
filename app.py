from dotenv import load_dotenv
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import pandas as pd
from helper import generate_create_table_statements,get_schema


load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
# llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)


def init_database(username, password, host, port, database):
    db_uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)



def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a PostgreSQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the PostgreSQL query and nothing else. Do not wrap the  query in any other text, not even backticks.
    
    IMP note: we are using a postgresql database so you must use postgresql syntax in your queries dont forget to use double quotes where needed and table name is case sensitive.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT "ArtistId", COUNT(*) AS track_count
        FROM "Track"
        GROUP BY "ArtistId"
        ORDER BY track_count DESC
        LIMIT 3;
        
    Question: Name 10 artists
    SQL Query: SELECT "Name" FROM "Artist" LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def get_schema_i(_):
        return get_schema(db)

    return (
        RunnablePassthrough.assign(schema= get_schema_i)
        | prompt
        | llm
        | StrOutputParser()
    )

def run_query(db, query):
    return db.run(query)

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema and db structure below, question, postgresql query, and postgresql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

    prompt = ChatPromptTemplate.from_template(template)



    chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=lambda _: get_schema(db),
        response=lambda vars: run_query(db,vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
    )

    return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
    })



if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Hello! I am an SQL asistant ask me anything about the database.")
    ]

st.set_page_config(page_title="talk 2 DB", page_icon=":rocket:")
st.title("Talk 2 DB")



with st.sidebar:
    st.write("Settings")
    st.write("This is a simple chat app that talks to a database.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("Database", value="chinook", key="Database")
    st.text_input("Username", value="dump", key="Username")
    st.text_input("Password", value="dump", type="password", key="Password")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["Username"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"],
            )
            st.session_state.db = db
            schema = get_schema(db)
            print(schema)
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    else:
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    if "db" in st.session_state:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            with st.spinner("Querying database..."):
                # time.sleep(10)
                # sql_chain = get_sql_chain(st.session_state.db)
                # response = sql_chain.invoke({
                #     "chat_history": st.session_state.chat_history,
                #     "question": user_query
                # })
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
    else:
        st.error("Please connect to a database first.")