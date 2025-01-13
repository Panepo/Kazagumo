from agent import agent

def demo():
  while 1:
    try:
      print("================================================")
      message = input("Please say something: ")

      if message == "exit" or message == "quit" or message == "":
          break

      response = agent.chat(message)
      print(response)
    except KeyboardInterrupt:
      break

if __name__ == "__main__":
  demo()
