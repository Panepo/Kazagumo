from agent import agent

while 1:
    print("================================================")
    text = input("Please say something: ")
    response = agent.chat(text)
