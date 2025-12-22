#!/usr/bin/env python
import sys
import warnings
import asyncio

from datetime import datetime

from crewai_tools.tools.stagehand_tool.stagehand_tool import asyncio

from crewai_trainig_step_by_step.crew import CrewaiTrainigStepByStep

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    # try:

    # asyncio.run(
    crew_app = CrewaiTrainigStepByStep()
    crew_app.crew().kickoff(inputs=inputs)

    task_output = crew_app.summarization_task()

    if task_output.pydantic:
        print("Pydantic Output:", task_output.pydantic)
        for k, v in task_output.pydantic.key_points:
            print(f"{k}: {v}")
    # except Exception as e:
    #     raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        CrewaiTrainigStepByStep().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewaiTrainigStepByStep().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        CrewaiTrainigStepByStep().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
