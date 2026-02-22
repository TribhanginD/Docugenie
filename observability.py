from prometheus_client import Counter, Histogram, Summary
import time

# Metrics definitions
QUERY_COUNT = Counter('docugenie_queries_total', 'Total number of queries')
RETRIEVAL_LATENCY = Histogram('docugenie_retrieval_latency_seconds', 'Retrieval phase latency')
GENERATION_LATENCY = Histogram('docugenie_generation_latency_seconds', 'Generation phase latency')
QUERY_COST = Counter('docugenie_query_cost_dollars_total', 'Total estimated cost per query')

class MetricsTracker:
    @staticmethod
    def track_query():
        QUERY_COUNT.inc()

    @staticmethod
    def time_retrieval():
        return RETRIEVAL_LATENCY.time()

    @staticmethod
    def time_generation():
        return GENERATION_LATENCY.time()

    @staticmethod
    def add_cost(amount: float):
        QUERY_COST.inc(amount)
