from typing import List

from crewai import LLM, Agent, Crew, LLMGuardrail, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel

from crewai_trainig_step_by_step.listener import CrewTrainingBus
from crewai_trainig_step_by_step.tools.serper_scraper_tool import SerperScrapeTool
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

from eval import EvalListenerSetup

from crewai_trainig_step_by_step.llm import asimov_llm, embedding_model

eval_listener = EvalListenerSetup()
training_listener = CrewTrainingBus()

class SummarizationOutput(BaseModel):
    topic: str
    summary: str
    key_points: List[str]


@CrewBase
class CrewaiTrainigStepByStep:
    """CrewaiTrainigStepByStep crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    llm: LLM = asimov_llm

    # @before_kickoff
    # def prepare_inputs(self, inputs):
    #     # Preprocess or modify inputs, that will be accessible in the 'inputs'
    #     # parameter of the crew. In our case, we have two inputs: topic and current_year.
    #     # Let's modify slightly the topic, by overriding it with an hardcoded option.
    #     # This method is often used to fetch external data such as s3 or any
    #     # other cloud bucket.
    #
    #     inputs["topic"] = "Agentic AI Stacks: Why CrewAI is the best option."
    #
    #     return inputs
    #
    # @after_kickoff
    # def log_results(self, result):
    #     # This method is often used to save the results to a file or any other storage.
    #     # More in general, it is used to perform any action after the crew has finished executing.
    #     # In this case, we are logging the results to the console.
    #     print("Crew execution completed with result:", result.json)
    #     return result.json

    def manager_agent(self) -> Agent:
        return Agent(
            role="manager",
            goal="Oversee the research and reporting process on the given topic.",
            backstory="You are an experienced project manager skilled in coordinating research and reporting tasks.",
            verbose=True,
            allow_delegation=True,
            reasoning=True,
            llm=self.llm,
        )



    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            llm=self.llm,
            tools=[SerperScrapeTool(), SerperDevTool()],
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],  # type: ignore[index]
            llm=self.llm,
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            async_execution=True,
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # type: ignore[index]
            output_file="report.md",
            async_execution=True,
        )

    @task
    def summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarization_task"],  # type: ignore[index]
            output_pydantic=SummarizationOutput,
            async_execution=False,
            guardrail=LLMGuardrail(
                description="Ensure the output follows the SummarizationOutput schema.",
                llm=self.llm,
            ),
            context=[
                self.reporting_task()
            ]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiTrainigStepByStep crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            manager_agent=self.manager_agent(),
            process=Process.hierarchical,
            verbose=True,
            output_log_file="crew_log.json",
            knowledge_sources=[TextFileKnowledgeSource(
                    file_paths=["user_preference.txt"],
                )
            ],
            tracing=True,
            memory=True,
            embedder=embedding_model
        )
