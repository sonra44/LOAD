from prometheus_client import Counter, Histogram, Gauge

INFERENCE_COUNT = Counter('ne_inference_total', 'Total inference calls')
INFERENCE_LATENCY = Histogram('ne_inference_duration_seconds', 'Inference latency')
ACTIVE_PROPOSALS = Gauge('ne_active_proposals', 'Number of active proposals generated')
AVG_CONFIDENCE = Gauge('ne_avg_confidence', 'Average confidence of proposals')
SAFETY_BLOCKS = Counter('ne_safety_blocks_total', 'Total safety blocks')
DEGRADATIONS = Counter('ne_degradation_to_rule', 'Total degradations to rule engine')
