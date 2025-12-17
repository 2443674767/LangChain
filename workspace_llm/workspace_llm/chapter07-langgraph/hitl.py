from typing import TypedDict, Annotated
import json
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import get_weather, write_file


# 定义 Agent 的状态结构
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 初始化本地大语言模型
llm = ChatOllama(
    # model="deepseek-r1:32b",
    model="llama3.1:8b",
    base_url="http://localhost:12356"
)

tools = [get_weather, write_file]
llm_with_tools = llm.bind_tools(tools)


# 聊天机器人节点
def chat_node(state: AgentState):
    messages = state["messages"]
    system_prompt = """你是一个智能助手，具备以下能力：
                    1. 查询天气信息
                    2. 结果写入文件
                    请根据用户的需求，选择合适的工具来完成任务。回答要准确、友好、专业。"""

    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + messages

    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


# 定义工具节点
tool_node = ToolNode(tools=tools)


# 动态路由：chat_node 之后
def route_after_chat(state: AgentState):
    """判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END


# 构建状态图
graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("tool_node", tool_node)

# 添加边
graph_builder.add_edge(START, "chat_node")
graph_builder.add_conditional_edges("chat_node", route_after_chat, ["tool_node", END])
graph_builder.add_edge("tool_node", "chat_node")

# 编译图结构 - 关键：使用 interrupt_before 在工具节点前中断
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tool_node"]  # 在执行工具前中断，等待人工确认
)

# 绘制可视化图表
graph.get_graph().draw_png('./graph4.png')


def run_with_approval():
    """运行带人工确认的工作流"""
    config = {"configurable": {"thread_id": "1"}}

    # 第一步：发送用户消息，执行到中断点
    print("【用户】北京天气怎么样\n")
    result = graph.invoke({"messages": ["北京天气怎么样"]}, config)

    # 检查是否中断（等待人工确认）
    snapshot = graph.get_state(config)

    if snapshot.next:  # 如果有下一个节点，说明被中断了
        print("工具调用需要人工确认")

        # 获取待执行的工具调用信息
        last_message = snapshot.values["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for idx, tool_call in enumerate(last_message.tool_calls, 1):
                print(f"\n[{idx}] 工具名称: {tool_call['name']}")
                print(f"    调用参数: {json.dumps(tool_call['args'], ensure_ascii=False, indent=4)}")

            approval = input("是否批准执行？(yes/no): ").strip().lower()

            if approval in ['yes', 'y']:
                print("工具调用已批准，继续执行...\n")
                # 继续执行（resume）
                result = graph.invoke(None, config)

            elif approval in ['no', 'n']:
                print("工具调用已被拒绝\n")
                # 手动添加拒绝消息，然后继续
                tool_messages = []
                for tool_call in last_message.tool_calls:
                    tool_messages.append(
                        ToolMessage(
                            content="工具调用被用户拒绝，请询问用户是否需要调整方案或提供更多信息。",
                            tool_call_id=tool_call["id"]
                        )
                    )
                # 更新状态并跳过工具节点
                graph.update_state(config, {"messages": tool_messages})
                result = graph.invoke(None, config)

    # 输出最终结果
    print("最终回复:")
    final_message = result["messages"][-1]
    print(final_message.content if hasattr(final_message, 'content') else str(final_message))


# 测试运行
if __name__ == "__main__":
    run_with_approval()