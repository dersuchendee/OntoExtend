
class MergeUnit:
    def __init__(self, solutions=[],strategy = "contatinate"):
  
        self.solutions = solutions
        self.strategy = strategy
        self.strategies = {
            "contatinate", # Decomposed prompting
            "union", # Ensemble prompting
            "select_best", # CoT-SC prompting
        }

    def merge(self, strategy = "contatinate"):
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if strategy == "contatinate":
            return self.contatinate(self.solutions)

        if strategy == "select_best":
            #ask llm which one is best
            #return the best
            #to do later
            pass

    def contatinate(self, solutions):
        """A simple merge for Decomposed prompting."""
        return '\n'.join(solutions)
        
