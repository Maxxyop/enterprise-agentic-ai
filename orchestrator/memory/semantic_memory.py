class SemanticMemory:
    def __init__(self):
        self.vulnerability_patterns = []

    def add_pattern(self, pattern):
        """Add a new vulnerability pattern to the memory."""
        self.vulnerability_patterns.append(pattern)

    def get_patterns(self):
        """Retrieve all stored vulnerability patterns."""
        return self.vulnerability_patterns

    def clear_memory(self):
        """Clear all stored patterns from memory."""
        self.vulnerability_patterns.clear()