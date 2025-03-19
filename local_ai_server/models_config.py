"""Model configurations and metadata."""

AVAILABLE_MODELS = {
    "phi-2.Q4_K_M.gguf": {
        "name": "Phi-2 (2.7B) Quantized",
        "description": "Microsoft's small but powerful model, great for low-end hardware",
        "size": "1.7GB",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "type": "gguf"
    },
    "tinyllama-1.1b-chat.Q4_K_M.gguf": {
        "name": "TinyLlama Chat (1.1B)",
        "description": "Extremely lightweight chat model, runs on most hardware",
        "size": "0.7GB",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "type": "gguf"
    },
    "neural-chat-7b-v3-1.Q4_K_M.gguf": {
        "name": "Neural Chat 7B v3.1",
        "description": "Balanced performance and quality, good for 8GB+ RAM",
        "size": "4.1GB",
        "url": "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf",
        "type": "gguf"
    },
    "stablelm-zephyr-3b.Q4_K_M.gguf": {
        "name": "StableLM Zephyr 3B",
        "description": "Fast and efficient 3B parameter model",
        "size": "1.9GB",
        "url": "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q4_K_M.gguf",
        "type": "gguf"
    },
    "openchat_3.5.Q4_K_M.gguf": {
        "name": "OpenChat 3.5",
        "description": "Open source alternative to ChatGPT",
        "size": "4.1GB",
        "url": "https://huggingface.co/TheBloke/openchat_3.5-GGUF/resolve/main/openchat_3.5.Q4_K_M.gguf",
        "type": "gguf"
    }
}
