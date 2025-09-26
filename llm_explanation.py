from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import json

load_dotenv()


def get_cause_llm(metric_metadata):
    client = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    prompt = """
        You are an expert in system monitoring and root cause analysis.
        Given the anomaly metadata below, suggest possible root causes.
        Metadata: {metadata}
        Return the response strictly in JSON format with fields:
        - root_cause
        - severity
        - suggested_action
        Also answer in short and simple in less words
    """

    template = PromptTemplate(template=prompt, input_variables=['metadata'])
    formatted_prompt = template.format(metadata=metric_metadata)

    print("\nExplaining the reason behind problem..............")
    raw_response = client.invoke(formatted_prompt).content

    try:
        cleaned = raw_response.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(cleaned)  
        # print(json.dumps(parsed, indent=2))  
        return parsed
    except json.JSONDecodeError:
        print("Model didn't return proper JSON, raw response shown:")
        # print(raw_response)
        return {"raw_response": raw_response}

