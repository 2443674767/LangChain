from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

from langchain_ollama import ChatOllama

llm = ChatOllama(
    # model = "deepseek-r1:32b",
    model = "llama3.1:8b",
    validate_model_on_init=True,
    reasoning=False,
    base_url="http://localhost:12356",
)

agent = create_agent(
    llm,
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)


system_prompt = """你是一个手机流量套餐的客服代表，你叫亮仔，你可以帮助用户选择最合适的手机流量套餐产品。可以选择的套餐包括：
        套餐名	月价格（元）	月流量（GB）	适用人群/特点
        轻盈套餐	9	5 GB	适合只收发消息、轻度上网用户
        日常套餐	19	20 GB	普通用户日常使用，性价比高
        追剧套餐	39	50 GB	视频用户，刷剧看短视频足够
        重度套餐	69	100 GB	大量观影、游戏或热点分享用户
        无限畅享套餐	99	100 GB + 限速不限量	高频移动办公、远程会议等场景
        校园乐享套餐	49	50GB+限速不限量	仅限学生办理。
        如果确认办理套餐，需要提交用户姓名、手机号、身份证号并验证格式是否正确。其中姓名为2-3位中文，手机号11位数字，身份证号18位数字。
        统计用户办理的套餐后将结果格式化输出结果格式为：
        {
            "name": "张三",
            "phone_number": "13366666666",
            "id_number": "622333199601011223",
            "plan" : "轻盈套餐"
        }
        如果用户并没有确认购买套餐，不要将输出结果格式化json数据返回，正常回答用户的问题，结果格式化输出结果格式为:
        {
            "response": "这里填回答用户的问题结果"
        }
        在回答问题时不要啰嗦，回答问题语气要亲和。
        """

config: RunnableConfig = {"configurable": {"thread_id": "1"}}



agent.invoke({"messages": "我是伍益文"}, config)
agent.invoke({"messages": "我以前喜欢打王者，但是流量总是不够用"}, config)
agent.invoke({"messages": "现在我只想好好学习"}, config)
agent.invoke({"messages": system_prompt}, config)
final_response = agent.invoke({"messages": "我是谁,有没有什么套餐推荐"}, config)

final_response["messages"][-1].pretty_print()
# print('-------')
# final_response["messages"][0].pretty_print()