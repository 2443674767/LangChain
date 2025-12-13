from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import time

# 初始化语言模型
chat_model = OllamaLLM(
    model = "deepseek-r1:70b",
    validate_model_on_init=True,
    reasoning=False,
    base_url="http://localhost:12356",
)

joke_query = "告诉我一个笑话。"

# 定义Json解析器
parser = JsonOutputParser()

# 以PromptTemplate为例
prompt_template = PromptTemplate.from_template(
    template="回答用户的查询\n 满足的格式为{format_instructions}\n 问题为{question}\n",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt_template | chat_model | parser

start_time = time.time()
json_result = chain.invoke(input={"question": joke_query})
duration = time.time() - start_time
print(f"调用耗时：{duration:.2f}秒")
print(json_result)