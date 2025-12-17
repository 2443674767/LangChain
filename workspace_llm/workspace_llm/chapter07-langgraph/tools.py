import json
import dotenv
from loguru import logger
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# 加载环境变量配置
dotenv.load_dotenv()


class WeatherQuery(BaseModel):
    """
    天气查询参数模型类，用于定义天气查询工具的输入参数结构。

    :param city: 城市名称，字符串类型，表示要查询天气的城市
    """
    city: str = Field(description="城市名称")


class WriteQuery(BaseModel):
    """
    写入查询模型类

    用于定义需要写入文档的内容结构，继承自BaseModel基类

    属性:
        content (str): 需要写入文档的具体内容，包含详细的描述信息
    """
    content: str = Field(description="需要写入文档的具体内容")


@tool(args_schema=WeatherQuery)
def get_weather(city):
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
            "main": {"temp": 18.2, "feels_like": 17.8, "temp_min": 18.2, "temp_max": 18.2, "pressure": 1010, "humidity": 72},
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


@tool(args_schema=WriteQuery)
def write_file(content):
    """
    将指定内容写入本地文件

    参数:
        content (str): 要写入文件的文本内容

    返回值:
        str: 表示写入操作成功完成的提示信息
    """
    # 将内容写入res.txt文件，使用utf-8编码确保中文字符正确保存
    with open('res.txt', 'w', encoding='utf-8') as f:
        f.write(content)
        logger.info(f"已成功写入本地文件，写入内容：{content}")
        return "已成功写入本地文件。"
