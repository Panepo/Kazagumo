import gradio as gr
from agent import agent

def chatbot(message, history):
  return agent.chat(message)

demo = gr.ChatInterface(fn=chatbot, type="messages", title="Chatbot")

if __name__ == "__main__":
  demo.launch()
