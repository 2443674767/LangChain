from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from tools import get_weather


# 定义 Agent 的状态结构，包含消息列表和审核状态
class AgentState(TypedDict):
    """
    描述 Agent 当前状态的数据结构。

    属性:
        messages (Annotated[list, add_messages]): 包含历史交互信息的消息列表，
                                                  使用 `add_messages` 合并新旧消息。
        user_approved (bool): 标记用户是否同意执行工具调用
    """
    messages: Annotated[list, add_messages]
    user_approved: bool


# 初始化本地大语言模型，配置基础URL、模型名称和推理模式
llm = ChatOllama(
    # model="deepseek-r1:32b",
    model="llama3.1:8b",
    base_url="http://localhost:12356"
)
tools = [get_weather]
model = llm.bind_tools(tools)


def call_model(state: AgentState):
    """
    调用绑定工具的大语言模型以生成响应。

    参数:
        state (AgentState): 包含当前会话中所有消息的状态对象。

    返回:
        dict: 新增模型响应后的更新状态（仅追加最新一条回复）。
    """
    system_prompt = SystemMessage("你是一个AI助手，可以依据用户提问产生回答，你还具备调用天气函数的能力")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# --- 人在闭环 (HITL) 节点 ---
def human_review(state: AgentState):
    """
    在执行工具调用之前请求人工审核确认。

    如果最后一条消息包含待执行的工具调用，则提示用户进行确认。
    若用户拒绝，则终止流程；否则允许进入工具执行阶段。

    参数:
        state (AgentState): 包含当前会话状态的对象。

    返回:
        dict: 根据用户选择决定下一步操作：
              - 用户拒绝时返回系统提示消息并标记为未批准；
              - 允许继续则标记为已批准。
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        call = last_message.tool_calls[0]
        tool_name = call["name"]
        tool_args = call["args"]

        print(f"[HITL] 模型计划调用工具 `{tool_name}`，参数：{tool_args}")
        confirm = input("[HITL] 是否确认执行？(y/n): ")

        if confirm.lower() != "y":
            # 用户拒绝，返回提示消息并标记为未批准
            return {
                "messages": [AIMessage(content="用户拒绝了工具调用，无法获取相关信息。")],
                "user_approved": False
            }

        # 用户同意，标记为已批准
        return {"user_approved": True}

    # 没有工具调用，直接标记为已批准
    return {"user_approved": True}


def should_review(state: AgentState):
    """
    判断是否需要进行人工审核。

    参数:
        state (AgentState): 当前状态

    返回:
        str: 下一个节点名称
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "human_review"
    return END


def should_execute_tools(state: AgentState):
    """
    判断是否应该执行工具。

    参数:
        state (AgentState): 当前状态

    返回:
        str: 下一个节点名称
    """
    if state.get("user_approved", False):
        return "tools"
    return END


# 创建工具节点，用于执行工具调用
tool_node = ToolNode(tools)

# 构建状态图结构
graph_builder = StateGraph(AgentState)

# 每个节点都与对应的处理函数进行绑定，构成工作流的基本单元
graph_builder.add_node("agent", call_model)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("tools", tool_node)

# 添加边：从 START 到 agent
graph_builder.add_edge(START, "agent")

# 添加条件边：根据是否有工具调用来判断是否需要人工审核
graph_builder.add_conditional_edges(
    "agent",
    should_review,
    {"human_review": "human_review", END: END}
)

# 添加条件边：根据用户是否同意来决定是否执行工具或结束
graph_builder.add_conditional_edges(
    "human_review",
    should_execute_tools,
    {"tools": "tools", END: END}
)

# 工具执行完成后重新回到 agent 继续对话循环
graph_builder.add_edge("tools", "agent")

# 创建内存保存器
memory = MemorySaver()

# 编译图结构，并绘制可视化图表
graph = graph_builder.compile(checkpointer=memory)
graph.get_graph().draw_png('./graph5.png')

# 配置对话线程ID
config = {"configurable": {"thread_id": "chat-1"}}

# 运行第一轮：问北京天气
print("\n" + "=" * 50)
print("第一轮对话：询问北京天气")
print("=" * 50)
response1 = graph.invoke({"messages": ["北京天气怎么样"]}, config=config)

print("\n=== 第一次结果 ===")
print(response1["messages"][-1].content)

# 打印已保存的检查点
print("\n" + "=" * 50)
print("检查点历史")
print("=" * 50)
states = list(graph.get_state_history(config))

for i, state in enumerate(states):
    print(f"\n=== 检查点 {i} (next: {state.next}) ===")
    print(f"Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
    if state.values.get("messages"):
        print(f"Messages count: {len(state.values['messages'])}")

# 从第二个检查点恢复并注入新问题
print("\n" + "=" * 50)
print("第二轮对话：从检查点恢复并询问上海天气")
print("=" * 50)
new_config = graph.update_state(
    states[1].config,
    values={"messages": [{"role": "user", "content": "上海天气怎么样"}]}
)

response2 = graph.invoke(None, config=new_config)

print("\n=== 第二次结果 ===")
print(response2["messages"][-1].content)