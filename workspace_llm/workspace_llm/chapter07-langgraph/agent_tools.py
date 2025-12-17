from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from tools import get_weather, write_file


# 定义 Agent 的状态结构，包含消息列表
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 初始化本地大语言模型，配置模型名称和推理模式
llm = ChatOllama(
    # model="deepseek-r1:32b",
    model="llama3.1:8b",
    base_url="http://localhost:12356"
)
tools = [get_weather, write_file]
llm_with_tools = llm.bind_tools(tools)


# 聊天机器人节点，用于处理对话状态并生成回复，并告诉模型可以调用哪些工具
def chat_node(state: AgentState):
    messages = state["messages"]
    system_prompt = """你是一个智能助手，具备以下能力：
                    1. 查询天气信息
                    2. 结果写入文件
                    请根据用户的需求，选择合适的工具来完成任务。回答要准确、友好、专业。"""
    # 构建完整的消息列表（系统提示词 + 用户消息）,如果第一条消息不是系统消息，则添加系统提示词
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(
            content=system_prompt)] + messages
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


# 定义工具节点（系统预置 ToolNode 会自动解析 tool_calls）
tool_node = ToolNode(tools=tools)


# 动态路由：chat_node → tool_node 或 END
def route_after_chat(state: AgentState):
    """判断是否需要进入工具节点"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END


# 构建状态图结构
graph_builder = StateGraph(AgentState)

# 每个节点都与对应的处理函数进行绑定，构成工作流的基本单元
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("tool_node", tool_node)

# 添加边：从 START 到 chatbot，然后到 END
graph_builder.add_edge(START, "chat_node")
# 添加条件边：根据是否有工具调用来判断是否需要进入工具节点
graph_builder.add_conditional_edges("chat_node", route_after_chat, ["tool_node", END])
# 工具节点执行完后回到 chat_node，继续多轮对话
graph_builder.add_edge("tool_node", "chat_node")

# 编译图结构，并绘制可视化图表
graph = graph_builder.compile()
graph.get_graph().draw_png('./graph3.png')

response1 = graph.invoke({"messages": ["北京天气怎么样"]})

print(response1["messages"][-1].content)