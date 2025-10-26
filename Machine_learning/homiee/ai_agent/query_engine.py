from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

# Load TinyLlama once
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def llm_to_mongo(user_query: str) -> list:
    """
    Use TinyLlama to infer a MongoDB aggregation pipeline.
    """

    prompt = f"""You are an assistant that translates English questions into MongoDB aggregation pipelines.
Return only valid JSON array representing the pipeline. Example:
Q: show average price in Sydney
A: [{{"$match": {{"City": "Sydney"}}}}, {{"$group": {{"_id": null, "avg_price": {{"$avg": "$Price"}}}}}}]

Q: {user_query}
A:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the JSON-like pipeline from the LLM text
    match = re.search(r"\[(.|\n)*\]", text)
    if not match:
        return [{"$limit": 5}]
    try:
        pipeline = json.loads(match.group(0))
        if isinstance(pipeline, list):
            return pipeline
    except Exception:
        pass
    return [{"$limit": 5}]
