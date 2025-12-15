import gradio as gr
import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

os.makedirs(DATA_PATH, exist_ok=True)

embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def get_uploaded_files():
    if not os.path.exists(DATA_PATH):
        return "No documents uploaded yet."
    files = os.listdir(DATA_PATH)
    if not files:
        return "No documents uploaded yet."
    return "\n".join([f"📄 {f}" for f in files])

def process_file(files):
    if not files:
        return get_uploaded_files(), "⚠️ No file selected."
    
    new_files_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        destination = os.path.join(DATA_PATH, filename)
        shutil.copy(file_path, destination)
        
        try:
            loader = PyPDFLoader(destination)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            if chunks:
                db.add_documents(chunks)
                new_files_count += 1
        except Exception as e:
            return get_uploaded_files(), f"❌ Error: {str(e)}"

    return get_uploaded_files(), f"✅ Successfully processed {new_files_count} new file(s)!"

def chat_logic(message, history):
    if not message:
        return ""
    
    greetings = ["hi", "hello", "hey", "how are you"]
    if message.lower().strip() in greetings:
        return "Hello! I am your RAGbot. Ready to analyze your documents."

    try:
        results = db.similarity_search_with_score(message, k=3)
    except:
        return "⚠️ Database Error: Please upload a document first."

    if not results:
        return "I couldn't find any relevant information."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    template = """
    You are a helpful AI assistant.
    1. Answer using ONLY the context below.
    2. If unsure, say "I don't know based on the document."
    
    Context:
    {context}
    
    ---
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    final_prompt = prompt.format(context=context_text, question=message)
    
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        response = model.invoke(final_prompt)
        return response.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

chatgpt_css = """
.gradio-container {
    background-color: #343541 !important;
    color: white !important;
}
.sidebar-col {
    background-color: #202123 !important;
    padding: 20px !important;
    border-right: 1px solid #4d4d4f;
}
.chatbot {
    background-color: #343541 !important;
    border: none !important;
    height: 700px !important;
}
.user-message {
    background-color: #343541 !important;
    border-bottom: 1px solid #4d4d4f;
}
.bot-message {
    background-color: #444654 !important;
    border-bottom: 1px solid #4d4d4f;
}
textarea, input {
    background-color: #40414f !important;
    border: 1px solid #303139 !important;
    color: white !important;
    border-radius: 5px !important;
}
button.primary {
    background-color: #10a37f !important;
    border: none !important;
    color: white !important;
}
"""

base_theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
)

with gr.Blocks(title="RAGbot") as demo:
    
    with gr.Row(elem_id="main-row"):
        
        with gr.Column(scale=1, elem_classes="sidebar-col"):
            gr.Markdown("### 📂 New Chat / Uploads")
            
            file_upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
            upload_btn = gr.Button("Process Documents", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False, show_label=False)
            gr.Markdown("### 🕒 History")
            file_list = gr.Textbox(value=get_uploaded_files(), show_label=False, interactive=False, lines=15)

        with gr.Column(scale=4):
            gr.Markdown("# RAGbot \n *The AI that knows what you've got.*")
            
            chatbot = gr.Chatbot(label="", show_label=False, height=650)
            
            with gr.Row():
                msg = gr.Textbox(scale=8, show_label=False, placeholder="Send a message...", container=False)
                submit_btn = gr.Button("➤", scale=1, variant="primary")
            
            with gr.Row():
                 clear_btn = gr.Button("Clear Conversation", variant="secondary", size="sm")

    upload_btn.click(process_file, [file_upload], [file_list, upload_status])

    def respond(message, chat_history):
        if chat_history is None: chat_history = []
        
        bot_message = chat_logic(message, chat_history)
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    print("Starting ChatGPT Style RAGbot...")
    demo.launch(theme=base_theme, css=chatgpt_css)