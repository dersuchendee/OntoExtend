from Components import CallLLMUnit, CallRAGUnit, MergeUnit, MemoryUnit, SplitUnit
from Components.PromptTemplate.DecomposedPromptingLong import Prompt as DecomposedPromptingLongPrompt


class DecomposedPrompting:
    def __init__(self, Stories=[], CQs=[], CoreOntology= '', rag_topK=20, rag_searchK=10000,
                 LLM='GPT-5', prompt_template = DecomposedPromptingLongPrompt,
                 split_type='Decomposed', merge_strategy='contatinate',
                 dependency = 'Independent'):
        # Initialize components
        self.MemoryU    = MemoryUnit.MemoryUnit()
        self.CallLLMU   = CallLLMUnit.LLMCaller()#.LLMCaller(prompt='',LLM=LLM)
        self.CallRAGU   = CallRAGUnit.RAGCaller()
        self.SplitU     = SplitUnit.SplitUnit(split_type=split_type)
        self.MergeU     = MergeUnit.MergeUnit()
        self.Prompt     = prompt_template
        self.Dependency = dependency  # 'Independent' or 'Dependent', used to say CQ2 relies on CQ1 or not
        self.CQs        = CQs
        self.Stories    = Stories
        self.rag_topK   = rag_topK
        self.rag_searchK= rag_searchK
        self.merge_strategy = merge_strategy
        # Initial RAG call to set up core ontology
        if CoreOntology == '':
            try:
                self.CoreOntology = self.CallRAGU.CallRAG(CQ=CQs[0],# "core?"
                                                        topK=self.rag_topK,
                                                        searchK=self.rag_searchK)
            except:
                print("RAG call failed, setting CoreOntology to empty string.")
                self.CoreOntology = CoreOntology
            
            self.MemoryU.UpdateCoreOntology(new_core=self.CoreOntology)
        else:
            self.CoreOntology = CoreOntology

        self.MemoryU.UpdateCoreOntology(new_core=self.CoreOntology)


    def process(self,topK=20, searchK=10000):

        #step 1: Generate the tasks from user query
        self.SplitU = SplitUnit.SplitUnit( split_type='Decomposed')
        Tasks = self.SplitU.Decomposer(CQs,Stories,self.Prompt,
                                          split_type='Decomposed',Dependency=self.Dependency,
                                          topK=self.rag_topK,searchK=self.rag_searchK)
        self.MemoryU.Tasks = Tasks


        #step 2: Solving the tasks
        for task in self.MemoryU.Tasks:
            solution = self.CallLLMU.CallLLM(prompt=task,LLM='GPT-5')
            self.MemoryU.MarkTaskDone(solution)


        #step 3: Merge the solutions
        self.MergeU.solutions = self.MemoryU.SolvedTasks
        final_solution = self.MergeU.merge(strategy=self.merge_strategy)
        print("=======================================")
        print("Final Solution:")
        print(final_solution)
        print("=======================================")

        self.MemoryU.PrintStatus()

if __name__ == "__main__":
    Stories = ["Story about AI and its applications.","Story about AI and its applications."]
    CQs = ["CQ1",'CQ2']
    DecomposedPromptingInstance = DecomposedPrompting(Stories=Stories, CQs=CQs, CoreOntology='')
    DecomposedPromptingInstance.process(topK=20, searchK=10000)
