import fire

from src.evaluator import Evaluator
from src.main import Main
from src.filtering import Filtering
# from src.pseudo_retriever import PseudoRetriever

if __name__ == "__main__":
    fire.Fire(
        {
            "main": Main,
            "eval": Evaluator,
            "filtering": Filtering,
            # "retrieve": PseudoRetriever,
        }
    )
