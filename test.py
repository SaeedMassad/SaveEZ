from llama_cpp import Llama

# Load the model
model = Llama(model_path="/Users/saeedmassad/Desktop/Honours Project/models/llama-2-7b.Q5_K_M.gguf")

response = model(
    "What is the best method for saving money?",
    max_tokens=200,  # Increase to allow a longer response
    temperature=0.8,  # Adjust creativity
)
print(response['choices'][0]['text'])

