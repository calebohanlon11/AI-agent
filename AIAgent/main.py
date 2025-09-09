from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search, reddit_search_api,reddit_post_retrieval
from prompts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_url_analysis_messages,
    get_synthesis_messages
)
import time, requests

def safe_download(url, headers, retries=3, delay=2):
    for attempt in range(retries):
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 400:
            print("Snapshot not ready yet (400), retrying...")
            time.sleep(delay)
        else:
            r.raise_for_status()
    raise RuntimeError("Failed after retries")

load_dotenv()


llm = init_chat_model("gpt-4o")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None

class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(description="List of Reddit URLS that contain valuable information for answering the users questions")



def google_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Google for: {user_question}")

    google_results = serp_search(user_question, engine="google")
    print(google_results)

    return {"google_results": google_results}





def bing_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching bing for: {user_question}")

    bing_results = serp_search(user_question, engine="bing")
    print(bing_results)


    return {"bing_results": bing_results}




def reddit_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Reddit for: {user_question}")

    reddit_results = reddit_search_api(keyword=user_question)
    print(f"Searching Reddit for: {user_question}")

    return {"reddit_results": reddit_results}



def analyse_reddit_posts(state: State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        return {"selected_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results)

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls

        print("Selected URLs:")
        for i, url in enumerate(selected_urls, 1):
            print(f"{i}. {url}")

    except Exception as e:
        print(e)
        selected_urls = []

    return{"selected_reddit_urls": selected_urls}



def retrieve_reddit_posts(state: State):
    print("Getting Reddit posts")

    selected_urls = state.get("selected_reddit_urls", [])

    if not selected_urls:
        return {"reddit_post_data": []}

    print(f"Processing {len(selected_urls)} Reddit posts")

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if reddit_post_data:
        print(f"Succussfuly got {len(reddit_post_data)} Reddit posts")
    else:
        print("Failed to get Reddit posts")
        reddit_post_data = []

    print(reddit_post_data)
    return {"reddit_post_data": reddit_post_data}








def analyse_google_results(state: State):
    print("Analysing Google search results")

    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")

    messages = get_google_analysis_messages(user_question, google_results)
    reply = llm.invoke(messages)

    return {"google_analysis": reply.content}




def analyse_bing_results(state: State):
    print("Analysing bing search results")

    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")

    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = llm.invoke(messages)

    return {"bing_analysis": reply.content}




def analyse_reddit_results(state: State):
    print("Analysing reddit search results")

    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")

    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = llm.invoke(messages)

    return {"reddit_analysis": reply.content}




def synthesise_analyses(state: State):
    print("Combine all results together")

    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis= state.get("reddit_analysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis, bing_analysis, reddit_analysis
    )

    reply = llm.invoke(messages)
    final_answer = reply.content

    return {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}





graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyse_reddit_posts", analyse_reddit_posts)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("analyse_google_results", analyse_google_results)
graph_builder.add_node("analyse_bing_results", analyse_bing_results)
graph_builder.add_node("analyse_reddit_results", analyse_reddit_results)
graph_builder.add_node("synthesise_analyses", synthesise_analyses)

graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")

graph_builder.add_edge("google_search", "analyse_reddit_posts")
graph_builder.add_edge("bing_search", "analyse_reddit_posts")
graph_builder.add_edge("reddit_search", "analyse_reddit_posts")
graph_builder.add_edge("analyse_reddit_posts", "retrieve_reddit_posts")

graph_builder.add_edge("retrieve_reddit_posts", "analyse_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyse_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyse_reddit_results")

graph_builder.add_edge("analyse_google_results", "synthesise_analyses")
graph_builder.add_edge("analyse_bing_results", "synthesise_analyses")
graph_builder.add_edge("analyse_reddit_results", "synthesise_analyses")

graph_builder.add_edge("synthesise_analyses", END)

graph = graph_builder.compile()



def run_chatbot():
    print("Multi-Source Research Agent")
    print("Type 'exit to quit\n'")

    while True:
        user_input = input("Ask me anything: ")
        if user_input.lower() == "exit":
            print("Goodbye")
            break

        state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_question": user_input,
            "google_results": None ,
            "bing_results": None,
            "reddit_results": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "bing_analysis": None,
            "reddit_analysis": None ,
            "final_answer": None,

        }

        print("\n Starting parallel research process...")
        print("Launching Google, Bing, and Reddit searches...\n")
        final_state = graph.invoke(state)

        if final_state.get("final_answer"):
            print(f"\nFinal answer:\n{final_state.get('final_answer')}\n")

        print( "-" * 80)

if __name__ == "__main__":
    run_chatbot()








