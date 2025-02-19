import ollama
import pprint

image_path = "./datasets/custom_dataset/images/train/img_bottle_1739976622293.jpg"

response = ollama.chat(
    model="llama3.2-vision",
    messages=[
        {
            "role": "user",
            "content": "This image is of John. What is he doing and what is he detected as?",
            "images": [image_path],
            "stream": True,
        }
    ],
)
pprint.pprint(response)
