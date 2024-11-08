from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
import uvicorn
import base64
from io import BytesIO
from PIL import Image
import requests
import time

# Initialize FastAPI app
app = FastAPI(title="LLaMA Vision API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor
model_id = "/root/autodl-tmp/Llama-3.2-11B-Vision-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Create pipeline without specifying device
pipe = pipeline(
    "image-to-text",
    model=model,
    processor=processor
)


# Define request models
class ImageURL(BaseModel):
    url: str


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[str, dict]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7


class Choice(BaseModel):
    message: Message
    finish_reason: str
    index: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]


def process_image(image_url: str) -> str:
    try:
        # Download image if URL provided
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Generate description
        result = pipe(image)[0]["generated_text"]
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Extract the last message content
        last_message = request.messages[-1].content

        # Handle different message formats
        if isinstance(last_message, list):
            # Process multimodal content
            prompt = ""
            for content in last_message:
                if isinstance(content, dict) and content.get("type") == "image_url":
                    image_url = content["image_url"]["url"]
                    image_description = process_image(image_url)
                    prompt += f"\nImage description: {image_description}\n"
                else:
                    prompt += str(content)
        else:
            prompt = str(last_message)

        # Generate response
        response = pipe(prompt)
        generated_text = response[0]["generated_text"]

        return ChatCompletionResponse(
            id="chatcmpl-" + base64.b64encode(str(hash(generated_text)).encode()).decode()[:10],
            object="chat.completion",
            created=int(time.time()),
            model=model_id,
            choices=[
                Choice(
                    message=Message(role="assistant", content=generated_text),
                    finish_reason="stop",
                    index=0
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


if __name__ == "__main__":
    print(f"Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)