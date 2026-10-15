'''Backward-compatible import path for the MGR retrieval implementation.'''
from layers.MGRRetriever import MGRRetriever

# Preserve imports used by older SARAF code and legacy model modules.
RetrievalTool = MGRRetriever
