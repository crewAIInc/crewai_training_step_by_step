import os
from typing import Any, Dict, List, Tuple, Union

from crewai import LLM, Agent, Crew, Process, Task, TaskOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


def validate_content_length(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate that the content meets requirements."""
    try:
        # Extract the actual content string from TaskOutput
        content = result.raw if hasattr(result, "raw") else str(result)

        # Check word count
        word_count = len(content.split())
        if word_count > 800:
            return (
                False,
                "Blog content exceeds 800 words, it should be less than 800 words",
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

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            from_repository="reporting-analyst",
            # knowledge_sources=[self.text_source],
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
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
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # memory=True,
            llm=self.llm,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
