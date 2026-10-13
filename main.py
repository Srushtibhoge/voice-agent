import asyncio
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

import pyttsx3

engine = pyttsx3.init()

load_dotenv()

# client = OpenAI()
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# async_client = AsyncOpenAI()

# async def tts(speech:str):
#     async with async_client.audio.speech.with_streaming_response.create(
#         model="gpt-4o-mini-tts",
#         voice="coral",
#         instructions="Always speak in cheerfull manner with full delight and happy",
#         input=speech,
#         responce_format = "pam",
#     ) as response:
#         await LocalAudioPlayer().play(response)

engine = pyttsx3.init()

SYSTEM_PROMPT = f"""
    You're an expert voice agent. You are give the transcript of what user
    has said using voice.
    You need to output as if you are an voice agent and whatever you speak
    will be converted back to audio using AI and played back to user.
"""

def main():
    r = sr.Recognizer() #Speech to Text
    
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    with sr.Microphone() as source : #mic access
       
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2
        
        while True:
            try:
                print("Speak something...")
                
                audio = r.listen(source)
                
                print("processing Audio.. (STT)")
                
                stt = r.recognize_google(audio)
                
                print('You said',stt)
                
                # Stop when you say exit
                if stt.lower().strip() in ["exit", "quit", "stop"]:
                    engine.say("Goodbye!")
                    engine.runAndWait()
                    break
                
                messages.append({
                    "role": "user",
                    "content": stt
                })
                
                response = client.chat.completions.create(
                    # model = "gpt-4.1-mini",
                    # model="gemini-3.8-flash",
                    model="gemini-3.5-flash-lite",
                    messages= messages
                )
                
                answer = response.choices[0].message.content
                print("AI response:", answer)

                messages.append({
                    "role": "assistant",
                    "content": answer
                })

                engine.say(answer)
                engine.runAndWait()
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand your speech.")

            except sr.RequestError as e:
                print("Speech recognition error:", e)

            except Exception as e:
                print("Error:", e)
                
        

main()
        
        