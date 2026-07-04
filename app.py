import os
import subprocess
import sys


# Force immediate background installations to skip dependency hell loops
def install_packages():
    required_libs = [
        "llama-index-core==0.12.40",
        "llama-index-llms-groq==0.3.2",
        "langchain-community",
    ]
    for lib in required_libs:
        try:
            # Silent installation execution
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", lib]
            )
        except Exception as e:
            print(f"Installation skip/warning on package {lib}: {e}")


# Run the runtime extraction engine
install_packages()

# ==========================================
# Standard Framework Imports
import gradio as gr
from dotenv import load_dotenv

# LlamaIndex Core & Interface Setup Elements
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# LlamaIndex Type Adapter Bridge
from llama_index.core.embeddings import LangchainEmbedding

# LangChain Community Embedding Engine Wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================

load_dotenv()

# --- 1. Initialize RAG Backend ---
model = "llama-3.3-70b-versatile"

# Secure API token acquisition
groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY ")
llm = Groq(model=model, token=groq_key)

# Initialize standard universal embedding engine wrapper
langchain_embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Wrap it in the LlamaIndex adapter class so the type assertion passes perfectly
embed_model = LangchainEmbedding(langchain_embed)

# Assign components to LlamaIndex Settings global workspace configurations
Settings.llm = llm
Settings.embed_model = embed_model

# Load your local vector registry maps safely
storage_context = StorageContext.from_defaults(persist_dir="vector_index")
vector_index = load_index_from_storage(storage_context)
retriever = vector_index.as_retriever(similarity_top_k=2)

prefix_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You are a nice chatbot having a conversation with a human.",
    ),
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="Answer the question based only on the following context and previous conversation.",
    ),
    ChatMessage(
        role=MessageRole.SYSTEM, content="Keep your answers short and succinct."
    ),
]

memory = ChatMemoryBuffer.from_defaults()

rag_bot = ContextChatEngine(
    llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
)


# --- 2. Gradio Interaction Callback ---
def custom_rag_bot_callback(message, history, top_k_value):
    rag_bot._retriever._similarity_top_k = int(top_k_value)
    rag_bot.reset()

    bot_history = []
    for m in history:
        # Prevent errors from reading greeting strings format variations
        if isinstance(m, dict) and "content" in m:
            content_text = m["content"]
            if isinstance(content_text, list) and len(content_text) > 0:
                content_text = content_text[0].get("text", "")
            bot_history.append(ChatMessage(role=m["role"], content=str(content_text)))
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            bot_history.append(ChatMessage(role="user", content=str(m[0])))
            bot_history.append(ChatMessage(role="assistant", content=str(m[1])))

    response = rag_bot.chat(message, chat_history=bot_history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response.response})

    return history


# --- 3. Gradio Layout & Interface Configuration ---
with gr.Blocks(title="Animal Farm Novel Analyzer") as demo_custom:
    gr.Markdown("<h1>The Animal Farm Novel: Political Satire Analyzer</h1>")
    gr.Markdown(
        "<p><em>An AI-powered RAG system designed to parse, retrieve, and analyze the deep allegorical layers of George Orwell's classic novel.</em></p>"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("## RAG Settings")
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=2,
                label="Number of Documents (Top K)",
                info="Controls how many relevant chunks of the novel are retrieved to answer your question.",
            )

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages",
                label="Orwellian Insights Engine",
                height=500,
                value=[
                    {
                        "role": "assistant",
                        "content": "Welcome! This system analyzes George Orwell's 'Animal Farm'—a novel using farm animals to satirize social and political revolutions. Ask me about the characters (Napoleon, Snowball, Boxer) or how the laws of the farm change!",
                    }
                ],
            )

            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something like: 'What is the real-world meaning behind Napoleon and Snowball's conflict?'",
            )

            clear_btn = gr.ClearButton([msg_input, chatbot], value="Reset Conversation")

            msg_input.submit(
                fn=custom_rag_bot_callback,
                inputs=[msg_input, chatbot, top_k_slider],
                outputs=[chatbot],
            ).then(fn=lambda: None, inputs=None, outputs=[msg_input], queue=False)

if __name__ == "__main__":
    demo_custom.launch()
