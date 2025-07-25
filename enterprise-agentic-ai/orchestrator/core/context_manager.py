class ContextManager:
    def __init__(self):
        self.context = {}

    def update_context(self, key, value):
        """Update the context with a new key-value pair."""
        self.context[key] = value

    def get_context(self, key):
        """Retrieve a value from the context by key."""
        return self.context.get(key, None)

    def clear_context(self):
        """Clear the entire context."""
        self.context.clear()

    def get_full_context(self):
        """Return the entire context."""
        return self.context.copy()