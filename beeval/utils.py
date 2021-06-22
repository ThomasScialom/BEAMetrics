import logging
logging.basicConfig(level=logging.CRITICAL)

logger = logging.getLogger()
component_logger = logger.getChild("BEAMetrics")
component_logger.setLevel(logging.INFO)