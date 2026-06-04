import os
import json
from pathlib import Path
from typing import List, Dict
import requests
import base64
import numpy as np
import pandas as pd
# from aliyunsdkcore.client import AcsClient
# from aliyunsdkcore.request import CommonRequest
import nls
from gtts import gTTS
from elevenlabs import ElevenLabs

from mm_story_agent.base import register_tool


# Due to the trouble regarding environment, we use dashscope to deploy and call the API for CosyVoice.
# class CosyVoiceSynthesizer:

#     def __init__(self) -> None:
#         self.access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
#         self.access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
#         self.app_key = os.environ.get('ALIYUN_APP_KEY')
#         self.setup_token()

#     def setup_token(self):
#         client = AcsClient(self.access_key_id, self.access_key_secret,
#                            'cn-shanghai')
#         request = CommonRequest()
#         request.set_method('POST')
#         request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
#         request.set_version('2019-02-28')
#         request.set_action_name('CreateToken')

#         try:
#             response = client.do_action_with_exception(request)
#             jss = json.loads(response)
#             if 'Token' in jss and 'Id' in jss['Token']:
#                 token = jss['Token']['Id']
#                 self.token = token
#         except Exception as e:
#             import traceback
#             raise RuntimeError(
#                 f'Request token failed with error: {e}, with detail {traceback.format_exc()}'
#             )

#     def call(self, save_file, transcript, voice="longyuan", sample_rate=16000):
#         writer = open(save_file, "wb")
#         return_data = b''

#         def write_data(data, *args):
#             nonlocal return_data
#             return_data += data
#             if writer is not None:
#                 writer.write(data)

#         def raise_error(error, *args):
#             raise RuntimeError(
#                 f'Synthesizing speech failed with error: {error}')

#         def close_file(*args):
#             if writer is not None:
#                 writer.close()

#         sdk = nls.NlsStreamInputTtsSynthesizer(
#             url='wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1',
#             token=self.token,
#             appkey=self.app_key,
#             on_data=write_data,
#             on_error=raise_error,
#             on_close=close_file,
#         )

#         sdk.startStreamInputTts(voice=voice, sample_rate=sample_rate, aformat='wav')
#         sdk.sendStreamInputTts(transcript,)
#         sdk.stopStreamInputTts()


@register_tool("gtts")
class GoogleTTSAgent:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def call(self, params):
        pages = params["pages"]
        save_path = params["save_path"]
        save_path = Path(save_path)
        
        for idx, page in enumerate(pages):
            tts = gTTS(text=page, lang='en')
            tts.save(str(save_path / f"p{idx + 1}.wav"))
        
        return {"modality": "speech"}
    

@register_tool("elevenlabs")
class ElevenLabsAgent:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
        

    def text_to_speech(self, text,lang, output_path="output.wav"):
        # audio_stream  = self.client.text_to_speech.convert(
        #     voice_id=self.voice_id,
        #     model_id="eleven_flash_v2_5",
        #     text=text,
        #     voice_settings={"stability": 0.75, "similarity_boost": 0.4, "speed": 1.2}
        # )
        if lang =="english":
            voice_id = "CoQByuTrT9gbKYx6QFL6"
        elif lang == "hindi":
            voice_id = "Sm1seazb4gs7RSlUVw7c"
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        headers = {
            "xi-api-key": os.getenv('ELEVENLABS_API_KEY'),
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75
            }
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            audio_data = base64.b64decode(result['audio_base64'])
            with open(output_path, "wb") as f:
                f.write(audio_data)
                print(f"✅ Audio saved to {output_path}")
            alignment = result.get("alignment", {})
            characters = alignment.get("characters", [])
            starts = alignment.get("character_start_times_seconds", [])
            ends = alignment.get("character_end_times_seconds", [])
            dtype = [('a', 'U10'), ('b', 'f4'), ('c', 'f4')]
            array = np.empty((1, 1,1), dtype=dtype)
            for char, start, end in zip(characters, starts, ends):
                new_triplet= np.array([[[(char, start, end)]]], dtype=dtype)
                array = np.append(array, new_triplet, axis=0)
            df = pd.DataFrame(array.reshape(-1))
            csv_name = output_path.with_suffix(".csv") 
            df.to_csv(csv_name, index=False)
            print(f"✅ timestamps saved to {csv_name}")
        else:
            raise Exception(f"Failed to convert text to speech: {response.status_code} {response.text}")

        # with open(output_path, "wb") as f:
        #     for chunk in audio_stream:
        #         f.write(chunk)

        return output_path
    
    def call(self, params):
        pages = params["pages"]
        save_path = params["save_path"]
        save_path = Path(save_path)
        lang = params["lang"]

        for idx, page in enumerate(pages):
            self.text_to_speech(text=page,lang=lang, output_path=save_path / f"p{idx + 1}.wav")

        return {"modality": "speech"}

