from langchain_community.llms import Ollama


query = "tell me a joke"
model = Ollama(
        model="qwen2:0.5b",
    )

print(model.invoke(query), end="")