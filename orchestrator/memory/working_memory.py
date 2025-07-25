class WorkingMemory:
    def __init__(self):
        self.context = {}

    def set_context(self, key, value):
        self.context[key] = value

    def get_context(self, key):
        return self.context.get(key, None)

    def clear_context(self):
        self.context.clear()