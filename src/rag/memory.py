class ConversationMemory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.history = []

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_history(self):
        formatted = ""
        for msg in self.history:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        return formatted
