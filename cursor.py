from openai import OpenAI;
from dotenv import load_dotenv
import requests
from typing import Optional
from pydantic import BaseModel, Field
import json
import os
import speech_recognition as sr


import pyttsx3

engine = pyttsx3.init()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def run_command(cmd: str):
    result = os.system(cmd)
    return result

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower}?format=%C+%t"
    response = requests.get(url);
    
    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong"

available_tools = {
    "get_weather": get_weather,
    "run_command":run_command
}

SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thought.
    You work on START, PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.
    You can also call a tool if required from the list of available tools.
    for every tool call wait for the observe step which is the output from the called tool
    
    Rules:
    - Strictly Follow the given JSON output format
    - Only run one step at a time.
    - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times)
    and finally OUTPUT (which is going to the  displayed to the user).
    
    
    Available Tools:
    - get_weather(city: str): Takes city name as an input string and returns the weather info about the city
    - run_command(cmd: str): Takes a system linux command as string and executes the command on user's system and returns the output from that come
    
    Output JSON format:
    { "steps": "START" | "PLAN" | "OUTPUT" | "TOOL". "content":"string", "tool": "string", "input": "string"}
    
    
    Examples 1:
    START: Hey, Can you solve 2+3*5/10
    PLAN: {"step": "PLAN": "content": "Seems like user is interested in math problem"}
    PLAN: {"step": "PLAN": "content": "looking at the problem, we should solve this using BODMOS method"}
    PLAN: {"step": "PLAN": "content": "Yes, The BODMAS is correct thing to be done here"}
    PLAN: {"step": "PLAN": "content": "first we must multiply 3 * 5 which is 15"}
    PLAN: {"step": "PLAN": "content": "Now the new equation is 2 + 15 /10"}
    PLAN: {"step": "PLAN": "content": "We must perform divide that is 15 / 10 = 1.5"}
    PLAN: {"step": "PLAN": "content": "Now the new equation is 2+1.5"}
    PLAN: {"step": "PLAN": "content": "Now finaaly lets perform the add 3.5}
    PLAN: {"step": "PLAN": "content": "Great, we have solved and finally left with 3.5 as ans"}
    OUTPUT: {"step": "OUTPUT" : "content:"3.5"}
    
     Examples 2:
        START: what is the weather of Delhi?
        PLAN: {'step": "PLAN": "content": "Seems like user is interested in getting weather of Delhi in India" }
        PLAN: { 'step": "PLAN": "content": "Lets see if we have any available tool from the list of available tools"}
        PLAN: { "step": "PLAN": "content": "Great, we have get_weather tool available for this query." }
        PLAN: { "step": "PLAN": "content": "I need to call get weather tool for delhi as input for city"}
        PLAN: { "step": "TOOL": "tool": "get_weather", "input": "delhi" }
        PLAN: { "step": "OBSERVE": "tool": "get_weather", "output": "The temp of delhi is cloudy with 20 C"}
        PLAN: { "step": "PLAN": "content": "Great, I got the weather info about delhi")
        OUTPUT: {"step": "OUTPUT" : "content": "The current weather in delhi is 20 C with some cloudy sky."} 
"""
message_history =[
    {"role":"system", "content":SYSTEM_PROMPT}
]

class MyOutputFormat(BaseModel):
    step: str = Field(..., description = "The ID of the step. Example PLAN, OUTPUT, TOOL. etc")
    content: Optional[str] = Field(None, description = "The optional string content for the step")
    tool: Optional[str] = Field(None,description = "The ID of the tool to call." )
    input: Optional[str] = Field(None,description = "The input params for the tool")
    
r = sr.Recognizer() #Speech to Text

with sr.Microphone() as source : #mic access

    r.adjust_for_ambient_noise(source)
    r.pause_threshold = 2
    
    while True:
        audio = r.listen(source)
                
        print("processing Audio.. (STT)")
        
        stt = r.recognize_google(audio)
        
        print('You said',stt)
        
        user_query = stt
        message_history.append({"role":"user", "content" : user_query})

        while True:
            responce = client.chat.completions.create(
            model="gpt-4o",
            # response_format={"type":"json_object"},
            response_format=MyOutputFormat,#formatted output
            messages= message_history
            )
            raw_result = responce.choices[0].message.content
            message_history.append({"role":"assistant", "content" : raw_result})
            
            # parsed_result = json.loads(raw_result)
            parsed_result = responce.choices[0].message.parsed#formatted output
            
            # if parsed_result.get("step") == "START":
            if parsed_result.step == "START":
                # print('start',parsed_result.get("content"))
                print('start',parsed_result.content)
                continue
            #below steps also need to make as parsed_result.step
            if parsed_result.get("step") == "TOOL":
                tool_to_call = parsed_result.get("tool")
                tool_input = parsed_result.get("input")
                print(f": {tool_to_call} ({tool_input})")
                tool_response = available_tools[tool_to_call](tool_input)
                print(f": {tool_to_call}({tool_input}) = (tool_response)")
                message_history.append({ "role": "developer", "content": json.dumps(
                { "step": "OBSERVE", "tool": tool_to_call, "input": tool_input, "output": tool_response})
                })
                continue
            
            if raw_result.step == "PLAN":
                print('plan',parsed_result.get("content"))
                continue
            
            if raw_result.step == "OUTPUT":
                print('end',parsed_result.get("content"))
                engine.say(parsed_result.get("content"))
                engine.runAndWait()
                break
                
                
                
                
            