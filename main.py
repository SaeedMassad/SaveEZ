import os
from llama_cpp import Llama
from query import handle_query  


def run_query():
    model = Llama(
        model_path="/Users/saeedmassad/Desktop/Honours Project/models/llama-2-7b.Q5_K_M.gguf",
        n_ctx=2048,
        n_threads=4
    )

    query = "summarize total money makeover by dave ramsey?"
    response = handle_query(query, chroma_db_path="./chroma_db", model=model)

    print(response)
    del model

run_query()
