import time
from agent import agent

def demo():
  while 1:
    print("================================================")
    text = input("Please say something: ")
    start_time = time.process_time()
    response = agent.chat(text)
    print(response)
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Process time: {elapsed_time} seconds")

if __name__ == "__main__":
  demo()
