class MemoryUnit:
    def __init__(self):
        # Initialize variables
        self.Tasks = []
        self.CoreOntology = ""
        self.CoreOntologies = [] #use this instead of coreontology if the tasks are independent or core is not updating
        self.SolvedPlans = []
        self.SolvedTasks = [] #solutions for decomposed prompting

    def UpdateCoreOntology(self, new_core=''''''):
        """Rewrite the CoreOntology variable from the RAG."""
        self.CoreOntology = new_core

    def UpdateTasks(self, new_tasks=[]):
        """list of all the tasks (or subtasks, plans)"""
        self.Tasks = new_tasks

    def MarkTaskDone(self, task):
        """Mark a task as done by moving it to SolvedTasks.
           We dont remove the task from the TODOs, we rather 
           check in length of the tasks that are done.
        """
        self.SolvedTasks.append(task)

    def MarkPlanDone(self, plan):
        """Mark a plan as done by adding it to SolvedPlans.
        it is final result in the CQbyCQ and CoT, but one
        of the results in CoT-SC
        """
        self.SolvedPlans.append(plan)

    def PrintStatus(self):
        print(f"CoreOntology: {self.CoreOntology[:50]}...")
        for index , task in enumerate(self.Tasks):
            print(f"Task {index+1}:")
            status = "Done" if index <= len(self.SolvedTasks)-1 else "Pending"
            print(f" - Task: {task[:80]+'''                 ...                 '''+task[-80:]}... Status: {status}")
            print('\n--------------------------------')
        print(f"SolvedPlans: {self.SolvedPlans}")
        print(f"SolvedTasks: {self.SolvedTasks}")