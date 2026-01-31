"""
MCP客户端测试用例
用于测试连接Go MCP服务和Python MCP服务
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger

# Python 3.11+ 支持 ExceptionGroup
try:
    from exceptiongroup import ExceptionGroup
except ImportError:
    # Python < 3.11 或未安装 exceptiongroup
    ExceptionGroup = None

# 加载 .env 文件中的环境变量
load_dotenv(override=True)

# 配置日志
logger.remove()  # 移除默认处理器
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "user-001"}}


def load_servers(file_path: str = "mcp.json") -> Dict[str, Any]:
    """
    从指定的 JSON 文件中加载 MCP 服务器配置。

    参数:
        file_path (str): 配置文件路径，默认为 "mcp.json"

    返回:
        Dict[str, Any]: 包含 MCP 服务器配置的字典，若文件中没有 "mcpServers" 键则返回空字典

    异常:
        FileNotFoundError: 如果配置文件不存在
        json.JSONDecodeError: 如果JSON格式错误
    """
    config_path = Path(file_path)

    if not config_path.exists():
        logger.error(f"❌ 配置文件不存在: {file_path}")
        logger.info("💡 提示: 请创建 mcp.json 文件，参考 mcp.json.example")
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            servers = data.get("mcpServers", {})

            if not servers:
                logger.warning("⚠️ 配置文件中没有找到 mcpServers 配置")
            else:
                logger.info(f"✅ 已加载 {len(servers)} 个MCP服务器配置: {list(servers.keys())}")

            return servers
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON格式错误: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ 加载配置文件失败: {e}")
        raise


def print_tool_info(tools: List[Any]) -> None:
    """
    打印工具信息

    参数:
        tools: 工具列表
    """
    if not tools:
        logger.warning("⚠️ 没有可用的工具")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"📋 可用工具列表 (共 {len(tools)} 个)")
    logger.info(f"{'='*60}")

    for i, tool in enumerate(tools, 1):
        tool_name = getattr(tool, 'name', 'Unknown')
        tool_desc = getattr(tool, 'description', '无描述')

        # 尝试获取工具的参数信息
        tool_args = ""
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
                properties = schema.get('properties', {})
                if properties:
                    arg_names = list(properties.keys())
                    tool_args = f"参数: {', '.join(arg_names)}"
            except:
                pass

        logger.info(f"{i:2d}. {tool_name}")
        logger.info(f"    描述: {tool_desc}")
        if tool_args:
            logger.info(f"    {tool_args}")

    logger.info(f"{'='*60}\n")


async def test_mcp_connection(servers_cfg: Dict[str, Any]) -> bool:
    """
    测试MCP服务器连接

    参数:
        servers_cfg: 服务器配置字典

    返回:
        bool: 连接是否成功
    """
    try:
        logger.info("🔌 正在连接MCP服务器...")
        mcp_client = MultiServerMCPClient(servers_cfg)
        tools = await mcp_client.get_tools()

        if tools:
            logger.info(f"✅ 连接成功！已加载 {len(tools)} 个工具")
            print_tool_info(tools)
            return True
        else:
            logger.warning("⚠️ 连接成功，但没有可用工具")
            return False

    except Exception as e:
        # 处理ExceptionGroup（Python 3.11+）或普通异常
        if ExceptionGroup and isinstance(e, ExceptionGroup):
            logger.error(f"❌ 连接MCP服务器失败: {e}")
            logger.error(f"   错误类型: ExceptionGroup (包含 {len(e.exceptions)} 个子异常)")
            for i, exc in enumerate(e.exceptions, 1):
                logger.error(f"   子异常 {i}: {type(exc).__name__}: {exc}")
                if hasattr(exc, '__cause__') and exc.__cause__:
                    logger.error(f"      原因: {exc.__cause__}")
        else:
            logger.error(f"❌ 连接MCP服务器失败: {e}")
            logger.error(f"   错误类型: {type(e).__name__}")
        import traceback
        logger.debug(f"   详细错误:\n{traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"❌ 连接MCP服务器失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")
        import traceback
        logger.debug(f"   详细错误:\n{traceback.format_exc()}")
        return False


async def run_chat_loop() -> None:
    """
    启动并运行一个基于 MCP 工具的聊天代理循环。

    该函数会：
    1. 加载 MCP 服务器配置；
    2. 初始化 MCP 客户端并获取工具；
    3. 创建基于 Ollama 的语言模型和代理；
    4. 启动命令行聊天循环；
    5. 在退出时清理资源。

    返回:
        None
    """
    logger.info("🚀 启动 MCP Agent 测试程序...")
    logger.info("="*60)

    # 1. 加载服务器配置
    try:
        servers_cfg = load_servers()
    except Exception as e:
        logger.error(f"❌ 无法加载服务器配置: {e}")
        return

    if not servers_cfg:
        logger.error("❌ 没有可用的服务器配置")
        return

    # 2. 测试连接并获取工具
    mcp_client = None
    try:
        logger.info("📡 初始化 MCP 客户端...")
        mcp_client = MultiServerMCPClient(servers_cfg)
        tools = await mcp_client.get_tools()

        if not tools:
            logger.error("❌ 没有获取到任何工具，请检查MCP服务器是否正常运行")
            return

        logger.info(f"✅ 已加载 {len(tools)} 个 MCP 工具")
        print_tool_info(tools)

    except Exception as e:
        # 处理ExceptionGroup（Python 3.11+）或普通异常
        if ExceptionGroup and isinstance(e, ExceptionGroup):
            logger.error(f"❌ 初始化MCP客户端失败: {e}")
            logger.error(f"   错误类型: ExceptionGroup (包含 {len(e.exceptions)} 个子异常)")
            for i, exc in enumerate(e.exceptions, 1):
                logger.error(f"   子异常 {i}: {type(exc).__name__}: {exc}")
                # 尝试获取更详细的错误信息
                if hasattr(exc, '__cause__') and exc.__cause__:
                    logger.error(f"      原因: {exc.__cause__}")
                # 如果是HTTP错误，显示状态码和URL
                if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
                    logger.error(f"      HTTP状态码: {exc.response.status_code}")
                    logger.error(f"      URL: {exc.response.url if hasattr(exc.response, 'url') else 'N/A'}")
                # 如果是连接错误，显示详细信息
                if 'Connection' in type(exc).__name__ or 'Connect' in str(type(exc)):
                    logger.error(f"      连接错误: 无法连接到服务器")
        else:
            logger.error(f"❌ 初始化MCP客户端失败: {e}")
            logger.error(f"   错误类型: {type(e).__name__}")

        import traceback
        logger.error(f"\n详细错误堆栈:\n{traceback.format_exc()}")

        # 提供解决建议
        logger.info("\n💡 可能的解决方案:")
        logger.info("   1. 检查Go MCP服务是否正在运行:")
        logger.info("      - 访问 http://localhost:9300/mcp 查看是否可访问")
        logger.info("      - 检查Go服务日志确认服务状态")
        logger.info("   2. 检查mcp.json配置:")
        logger.info("      - URL应该是 http://localhost:9300/mcp (注意是 /mcp 不是 /sse)")
        logger.info("      - transport应该是 \"streamable_http\" (字符串，不是字典)")
        logger.info("   3. 确认网络连接正常:")
        logger.info("      - 尝试用浏览器或curl访问 http://localhost:9300/mcp")
        logger.info("   4. 检查防火墙设置")
        return
    except Exception as e:
        logger.error(f"❌ 初始化MCP客户端失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"详细错误:\n{traceback.format_exc()}")

        # 提供解决建议
        logger.info("\n💡 可能的解决方案:")
        logger.info("   1. 检查Go MCP服务是否正在运行 (http://localhost:9300/mcp)")
        logger.info("   2. 检查mcp.json配置中的URL和transport类型是否正确")
        logger.info("   3. 确认网络连接正常")
        logger.info("   4. 查看Go MCP服务的日志以获取更多信息")
        return

    # 3. 初始化语言模型
    try:
        logger.info("🤖 初始化语言模型...")
        llm = ChatOllama(
            model="llama3.1:8b",
            base_url="http://localhost:12356",
            temperature=0.7,
        )
        logger.info("✅ 语言模型初始化成功")
    except Exception as e:
        logger.error(f"❌ 初始化语言模型失败: {e}")
        logger.error("   请确保Ollama服务正在运行，并且模型已下载")
        return

    # 4. 构建LangGraph Agent
    try:
        logger.info("🔧 构建 LangGraph Agent...")

        # 动态生成工具描述
        tool_descriptions = []
        for i, tool in enumerate(tools[:10], 1):  # 只显示前10个工具
            tool_name = getattr(tool, 'name', f'tool_{i}')
            tool_desc = getattr(tool, 'description', '无描述')
            tool_descriptions.append(f"{i}. {tool_name} - {tool_desc}")

        if len(tools) > 10:
            tool_descriptions.append(f"... 还有 {len(tools) - 10} 个工具")

        prompt = f"""你是一个智能助手，可以通过调用以下工具来帮助用户完成任务：

可用工具：
{chr(10).join(tool_descriptions)}

请根据用户的自然语言请求，判断是否需要调用工具。如果需要，请正确使用工具并返回结果。
如果不需要调用工具，就直接回答用户的问题。

重要提示：
- 仔细阅读每个工具的描述和参数要求
- 确保传递正确的参数类型和格式
- 如果工具调用失败，请向用户说明原因"""

        agent = create_react_agent(
            model=llm,
            prompt=prompt,
            tools=tools,
            checkpointer=checkpointer
        )
        logger.info("✅ Agent 构建成功")

    except Exception as e:
        logger.error(f"❌ 构建Agent失败: {e}")
        import traceback
        logger.debug(f"详细错误:\n{traceback.format_exc()}")
        return

    # 5. CLI聊天循环
    logger.info("\n" + "="*60)
    logger.info("🤖 MCP Agent 已启动，可以开始对话了！")
    logger.info("💡 输入 'quit' 或 'exit' 退出")
    logger.info("💡 输入 'tools' 查看可用工具列表")
    logger.info("💡 输入 'help' 查看帮助信息")
    logger.info("="*60 + "\n")

    try:
        while True:
            try:
                user_input = input("\n👤 你: ").strip()

                if not user_input:
                    continue

                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 再见！")
                    break

                if user_input.lower() == 'tools':
                    print_tool_info(tools)
                    continue

                if user_input.lower() == 'help':
                    logger.info("\n可用命令:")
                    logger.info("  - quit/exit/q: 退出程序")
                    logger.info("  - tools: 显示工具列表")
                    logger.info("  - help: 显示帮助信息")
                    logger.info("\n或者直接输入你的问题，Agent会尝试使用工具来回答。")
                    continue

                # 执行查询
                logger.info("🤔 正在思考...")
                result = await agent.ainvoke(
                    {"messages": [("user", user_input)]},
                    config
                )

                # 显示结果
                if result and 'messages' in result and result['messages']:
                    last_message = result['messages'][-1]
                    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    print(f"\n🤖 AI: {content}")
                else:
                    logger.warning("⚠️ 没有收到有效响应")

            except KeyboardInterrupt:
                logger.info("\n\n👋 收到中断信号，正在退出...")
                break
            except Exception as exc:
                logger.error(f"\n❌ 处理请求时出错: {exc}")
                import traceback
                logger.debug(f"详细错误:\n{traceback.format_exc()}")
                logger.info("💡 提示: 可以尝试重新输入问题，或输入 'quit' 退出")

    finally:
        # 6. 清理资源
        logger.info("\n🧹 正在清理资源...")
        try:
            if mcp_client:
                # 如果MultiServerMCPClient有close方法，调用它
                if hasattr(mcp_client, 'close'):
                    await mcp_client.close()
                elif hasattr(mcp_client, 'aclose'):
                    await mcp_client.aclose()
            logger.info("✅ 资源清理完成")
        except Exception as e:
            logger.warning(f"⚠️ 清理资源时出错: {e}")

        logger.info("👋 程序已退出，再见！")


async def main():
    """主函数"""
    try:
        await run_chat_loop()
    except KeyboardInterrupt:
        logger.info("\n\n👋 程序被用户中断")
    except Exception as e:
        logger.error(f"\n❌ 程序异常退出: {e}")
        import traceback
        logger.error(f"详细错误:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # 启动异步事件循环并运行聊天代理
    asyncio.run(main())
