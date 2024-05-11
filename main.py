import os
from dotenv import load_dotenv
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from crewai import Agent, Task, Crew

# load dotenv:
load_dotenv()

#Below are mandatory varbiles that shown by the author for simplifiying purpose
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-70b-8192'  # Adjust based on available model

# Initialize the Google Jobs tool
google_jobs_tool = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())

# Define your agents with roles and goals
recruitment_agent = Agent(
    role='Recruitment Agent',
    goal='Find suitable job candidates for various positions within the company',
    backstory="""You are a recruitment specialist with expertise in identifying and attracting top talent
  across various industries and roles.""",
    verbose=True,
    allow_delegation=True,
    tools=[google_jobs_tool]
)

recruitment_communicator = Agent(
    role='Recruitment Communications Coordinator',
    goal='Communicate job openings and company culture to potential applicants',
    backstory="""As an HR communications coordinator, you excel at crafting compelling job descriptions
  and showcasing the company's values and culture to attract the right candidates.""",
    verbose=True,
    allow_delegation=True,
)

# Create tasks for your agents
task1 = Task(
    description="""Identify current job openings in the field of software development using the Google Jobs tool.
  Focus on roles suitable for candidates with 1-3 years of experience.""",
    expected_output="Current job openings as a list",
    agent=recruitment_agent,
)

task2 = Task(
    description="""Based on the job openings identified, create engaging job descriptions and a recruitment
  social media post. Emphasize the company's commitment to innovation and a supportive work environment.""",
    expected_output="Job descriptions ready for social media as post",
    agent=recruitment_communicator,
)

# Instantiate your crew with a sequential process (by default tasks are executed sequentially)
crew = Crew(
    agents=[recruitment_agent, recruitment_communicator],
    tasks=[task1, task2],
    verbose=2
)

# Kick off the crew to start on it's tasks
result = crew.kickoff()

print("######################")
print(result)

