import evaluate


class SelfBLEU:
    def __init__(self) -> None:
        self.bleu = evaluate.load("bleu")

    def compute(self, candidates):
        predictions = []
        references = []
        for i in range(len(candidates)):
            predictions.append(candidates[i])
            references.append(candidates[:i] + candidates[i + 1 :])

        results = self.bleu.compute(predictions=predictions, references=references)

        return results["bleu"]
