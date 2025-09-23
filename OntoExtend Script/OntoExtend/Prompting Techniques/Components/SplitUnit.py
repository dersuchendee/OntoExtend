from Components import CallRAGUnit # import RAGCaller
class SplitUnit:
    def __init__(self, split_type= "Decomposed"):
        # Initialize variables
        self.SplitType = split_type  # 'Decomposed', 'CoT', 'CoT-SC'
        self.CQs = []  # Clarifying Questions
        self.Stories = []  # Narratives or task splits
        self.Plans = []# 2D set if Tasks only for CoT-SC, ToT and GoT
        self.Tasks = []# 1D set if Tasks

    def Decomposer(self, CQs,Stories,Prompt,Dependency='Independent', split_type = 'Decomposed',searchK=10000,topK=20):
        if self.SplitType == "Decomposed":
            if Dependency == 'Independent':
                try:
                    Cores = [ CallRAGUnit.RAGCaller.CallRAG(CQ=CQ,topK=topK,
                                                        searchK=searchK) for CQ in CQs]
                except:
                    print(f"RAG call failed for the CQ, using an empty string for all.")
                    Cores = ['' for CQ in CQs]
                self.Tasks = [Prompt.format(CQ=CQ, Story=Story, Ontology=Core) for CQ, Story,Core in zip(CQs, Stories,Cores)]
                return self.Tasks
            
            elif Dependency == 'Dependent': # later
                print("Dependent splitting not implemented yet.")
                return  1/0
            
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
