from crewai.events import (
    BaseEventListener,
    TaskCompletedEvent,
)
from crewai.task import Task
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator


class EvalListenerSetup(BaseEventListener):
    def __init__(self):
        super().__init__()
        # Store evaluation results directly as instance attributes
        self.quality_score = None
        self.suggestions = None

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source: Task, event: TaskCompletedEvent):
            if source.name == "reporting_task":
                evaluator = TaskEvaluator(source.agent)
                evaluation = evaluator.evaluate(source)

                print("evaluation quality:", evaluation.quality)
                print("evaluation suggestions:", evaluation.suggestions)

                # Store evaluation results directly on this instance
                self.quality_score = evaluation.quality
                self.suggestions = evaluation.suggestions
