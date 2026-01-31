import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient


def load_servers(path="mcp.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["mcpServers"]


async def main():
    # 1️⃣ 读取 MCP Server 配置
    servers = load_servers()

    # 2️⃣ 创建 MCP Client
    client = MultiServerMCPClient(servers)

    # 3️⃣ 拉取工具列表（最重要的一步）
    tools = await client.get_tools()

    print("\n✅ 已发现 MCP 工具：")
    for t in tools:
        print(f" - {t.name}")
        print(f"   desc: {t.description}")
        print(f"   schema: {t.args_schema}\n")

    # 4️⃣ 手动调用一个 Tool（按你的实际 Tool 名改）
    tool_name = "device.get_status"

    print(f"\n🚀 调用工具: {tool_name}")

    result = await client.call_tool(
        tool_name,
        arguments={
            "device_id": 1
        }
    )

    print("\n📦 工具返回结果：")
    print(result)

    # 5️⃣ 关闭连接
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
