import os
import gradio as gr
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

load_dotenv()

# --- 1. Initialize RAG Backend ---
model = "llama-3.3-70b-versatile"

# Look for GROQ_API_KEY securely from environment variables
groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY ")
llm = Groq(model=model, token=groq_key)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "./embedding_model/"

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model, cache_folder=embeddings_folder
)

# Load the local index folder you see in your file tree
storage_context = StorageContext.from_defaults(persist_dir="vector_index")
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
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
    print(history)

    bot_history = [
        ChatMessage(role=m["role"], content=m["content"][0]["text"]) for m in history
    ]

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
