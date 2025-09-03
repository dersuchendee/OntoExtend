class MemoryUnit:
    def __init__(self):
        # Initialize variables
        self.Tasks = []
        self.CoreOntology = ""
        self.SolvedPlans = []
        self.SolvedTasks = []

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
