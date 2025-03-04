class Similarity:
    @staticmethod
    def create(name, *args, **kwargs):
        if name == "constant":
            from src.similarity.constant import ConstantSimilarity

            return ConstantSimilarity(*args, **kwargs)
        elif name == "random":
            from src.similarity.random import RandomSimilarity

            return RandomSimilarity(*args, **kwargs)
        elif name == "mpnet":
            from src.similarity.mpnet import MPNetSimilarity

            return MPNetSimilarity(*args, **kwargs)
        else:
            raise ValueError(f"Unknown similarity model {name}")

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, target, reference):
        if isinstance(target[0], list):
            return self.batch_run(target, reference)
        else:
            return self.run(target, reference)

    def batch_run(self, targets, references):
        results = []
        for target, reference in zip(targets, references):
            results.append(self.run(target, reference))

        return results

    def run(self, target, reference):
        raise NotImplementedError
