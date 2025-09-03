class SplitUnit:
    def __init__(self, split_type= "Decomposed"):
        # Initialize variables
        self.SplitType = split_type  # 'Decomposed', 'CoT', 'CoT-SC'
        self.CQs = []  # Clarifying Questions
        self.Stories = []  # Narratives or task splits
        self.Plans = []# 2D set if Tasks only for CoT-SC, ToT and GoT
        self.Tasks = []# 1D set if Tasks

    def Decomposer(self, task: str, split_type: str = None):
        if self.SplitType == "Decomposed":
            self.Tasks = self.CQs
            return self.Plans, self.Tasks
        
        if self.SplitType == "CoT":
            #1- call LLM Unit with a CoT planner prompt
            #2- save the Plan
            #3- split the plan into tasks (explicitly asked from LLM to do)
            #4- similar to Decomposed, return the tasks
            pass

        if self.SplitType == "CoT-SC":
            #later
            pass

    def AddCQs(self, cqs):
        """add all CQs."""
        self.CQs = cqs

    def AddStories(self, story):
        self.Stories = story
