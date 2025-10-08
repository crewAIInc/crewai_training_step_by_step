import os
from typing import Any, Dict, List, Tuple, Union

from crewai import (
    LLM,
    Agent,
    Crew,
    LLMGuardrail,
    Process,
    Task,
    TaskOutput,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, task
from crewai_tools import SerperDevTool

from crewai_trainig_step_by_step.tools.custom_tool import SerperScrapeTool


def validate_content_length(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate that the content meets requirements."""
    try:
        # Extract the actual content string from TaskOutput
        content = result.raw if hasattr(result, "raw") else str(result)

        # Check word count
        word_count = len(content.split())
        if word_count > 1500:
            return (
                False,
                "Blog content exceeds 1500 words, it should be less than 1500 words",
            )

        # Additional validation logic here
        return (True, content.strip())
    except Exception as e:
        return (False, f"Unexpected error during validation: {str(e)}")


@CrewBase
class CrewaiTrainigStepByStep:
    """CrewaiTrainigStepByStep crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    llm = LLM(
        model=os.getenv("MODEL"),
        temperature=0.2,
    )

    text_source = TextFileKnowledgeSource(
        file_paths=[
            "user_preference.txt",
        ]
    )

    @before_kickoff
    def prepare_inputs(self, inputs):
        # Preprocess or modify inputs, that will be accessible in the 'inputs'
        # parameter of the crew. In our case, we have two inputs: topic and current_year.
        # Let's modify slightly the topic, by overriding it with an hardcoded option.
        # This method is often used to fetch external data such as s3 or any
        # other cloud bucket.

        inputs["topic"] = "Agentic AI Stacks: Why CrewAI is the best option."

        return inputs

    @after_kickoff
    def log_results(self, result):
        # This method is often used to save the results to a file or any other storage.
        # More in general, it is used to perform any action after the crew has finished executing.
        # In this case, we are logging the results to the console.
        print("Crew execution completed with result:", result)
        return result

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            tools=[SerperScrapeTool(), SerperDevTool()],
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            from_repository="reporting-analyst",
            knowledge_sources=[self.text_source],
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            guardrail=LLMGuardrail(
                description="The research task should be detailed and strictly related to the topic.",
                llm=LLM(model="gpt-4.1-mini", temperature=0.2),
            ),
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # type: ignore[index]
            output_file="report.md",
            guardrail=validate_content_length,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiTrainigStepByStep crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # memory=True,
            llm=self.llm,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
