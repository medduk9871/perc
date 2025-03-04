from src.similarity.similarity import Similarity
from sentence_transformers import SentenceTransformer, util


class MPNetSimilarity(Similarity):
    def __init__(self, *args, **kwargs):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def run(self, target, reference):
        reference_embedding = self.model.encode(reference)
        target_embedding = self.model.encode(target)
        scores = util.cos_sim(target_embedding, reference_embedding)

        return scores
