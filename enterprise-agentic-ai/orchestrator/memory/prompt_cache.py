from redis import Redis

class PromptCache:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis = Redis(host=redis_host, port=redis_port, db=redis_db)

    def cache_prompt(self, key, prompt):
        self.redis.set(key, prompt)

    def get_prompt(self, key):
        return self.redis.get(key)

    def delete_prompt(self, key):
        self.redis.delete(key)

    def clear_cache(self):
        self.redis.flushdb()