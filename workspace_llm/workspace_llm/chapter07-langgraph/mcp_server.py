import json
import os
import httpx
import dotenv
from mcp.server.fastmcp import FastMCP
from loguru import logger

dotenv.load_dotenv()

# 创建FastMCP实例，用于启动天气服务器SSE服务
mcp = FastMCP("WeatherServerSSE", host="0.0.0.0", port=8000)


@mcp.tool()
def get_weather(city: str):
    """
    查询指定城市的即时天气信息。

    :param city: 必要参数，字符串类型，表示要查询天气的城市名称。
                 注意：中国城市需使用其英文名称，如 "Beijing" 表示北京。
    :return: 返回 OpenWeather API 的响应结果，URL 为
             https://api.openweathermap.org/data/2.5/weather。
             响应内容为 JSON 格式的字符串，包含详细的天气数据。
    """

    # 发送 GET 请求并获取响应
    # response = httpx.get(url, params=params)
    if city.lower() == "beijing":
        response = {
            "coord": {"lon": 116.4074, "lat": 39.9042},
            "weather": [{"id": 801, "main": "Clouds", "description": "few clouds", "icon": "02d"}],
            "base": "stations",
            "main": {"temp": 12.5, "feels_like": 11.9, "temp_min": 12.5, "temp_max": 12.5, "pressure": 1012,
                     "humidity": 52},
            "visibility": 6000,
            "wind": {"speed": 3.6, "deg": 80},
            "clouds": {"all": 23},
            "dt": 1671673146,
            "sys": {"type": 1, "id": 9246, "country": "CN", "sunrise": 1671650426, "sunset": 1671693807},
            "timezone": 28800,
            "id": 1816670,
            "name": "Beijing",
            "cod": 200
        }

    elif city.lower() == "shanghai":
        response = {
            "coord": {"lon": 121.4737, "lat": 31.2304},
            "weather": [{"id": 802, "main": "Clouds", "description": "scattered clouds", "icon": "03d"}],
            "base": "stations",
            "main": {"temp": 18.2, "feels_like": 17.8, "temp_min": 18.2, "temp_max": 18.2, "pressure": 1010,
                     "humidity": 72},
            "visibility": 5000,
            "wind": {"speed": 4.1, "deg": 150},
            "clouds": {"all": 38},
            "dt": 1671673146,
            "sys": {"type": 1, "id": 9261, "country": "CN", "sunrise": 1671650044, "sunset": 1671692578},
            "timezone": 28800,
            "id": 1796236,
            "name": "Shanghai",
            "cod": 200
        }
    else:
        return {
            "error": "city_not_found",
            "message": "城市未找到，请输入有效城市名称"
        }

    # 将响应解析为 JSON 并序列化为字符串返回
    # data = response.json()
    # logger.info(f"查询天气结果：{json.dumps(data)}")
    logger.info(f"查询天气结果：{response}")
    return response


import asyncio

async def run_mcp():
    mcp.run(transport="sse")


if __name__ == "__main__":
    logger.info("启动 MCP SSE 天气服务器，监听 http://0.0.0.0:8000/sse")
    # 运行MCP客户端，使用Server-Sent Events(SSE)作为传输协议
    mcp.run(transport="sse")
    # asyncio.run(run_mcp())
