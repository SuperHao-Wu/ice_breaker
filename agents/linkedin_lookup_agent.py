from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        client="test",
        temperature=0,
        max_retries=3,
        model_name="gpt-3.5-turbo",
        verbose=True,
    )
    template = """Given the fullname {name_of_person} I want you to get me a link to their linkedin profile page. 
    Your answer should contain only a url.
    """

    tools_for_agent = [
        Tool(
            name="crawl google for linkedin profile page",
            func=get_profile_url,
            description="useful when you need get the linkedin URL",
        )
    ]
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    promt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    linked_profile_url = agent.run(promt_template.format_prompt(name_of_person=name))
    return linked_profile_url
