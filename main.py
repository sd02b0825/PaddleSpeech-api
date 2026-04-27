"""
PaddleSpeech 音频分类 API
基于 FastAPI 实现，提供音频分类检测接口
"""
import io
import wave
import os
import base64
import uuid
import subprocess
from datetime import datetime
import logging
import numpy as np
import noisereduce as nr
from scipy import signal
from typing import Optional
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
import time
from threading import Lock
import opuslib
import sherpa_onnx

# 缓存结构: {client_id: {"text": matched_text, "timestamp": time.time()}}
_command_cache = {}
_cache_lock = Lock()
_CACHE_TTL = 300  # 5分钟缓存有效期（秒）
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from paddlespeech.server.bin.paddlespeech_client import CLSClientExecutor
from paddlespeech.server.bin.paddlespeech_client import TTSClientExecutor
from http_client import CommandHttpClient
from audio_util import reduce_noise
from opus_decoder import OpusDecoder

# 初始化 logger
logger = logging.getLogger(__name__)

# 加载配置文件
def load_config():
    """加载 YAML 配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        return None

# 全局配置
CONFIG = load_config()

app = FastAPI(
    title="PaddleSpeech 音频分类 API",
    description="音频分类检测接口，支持传入音频文件路径进行分类检测",
    version="1.0.0"
)
# 启动配置
SERVER_PORT = CONFIG.get("server", {}).get("port", 8091) if CONFIG else 8091
SERVER_IP = CONFIG.get("server", {}).get("ip", "0.0.0.0") if CONFIG else "0.0.0.0"

# 从配置文件读取参数
DEFAULT_SERVER_IP = CONFIG.get("cls_config", {}).get("ip", "127.0.0.1") if CONFIG else "127.0.0.1"
DEFAULT_PORT = CONFIG.get("cls_config", {}).get("port", 8090) if CONFIG else 8090
DEFAULT_TOPK = CONFIG.get("cls_config", {}).get("topk", 1) if CONFIG else 1
TEMP_DIR = CONFIG.get("cls_config", {}).get("temp_dir", "/workspace/temp") if CONFIG else "/workspace/temp"
SAMPLE_RATE = CONFIG.get("cls_config", {}).get("sample_rate", 16000) if CONFIG else 16000
SCORE = CONFIG.get("cls_config", {}).get("score", 0.5) if CONFIG else 0.5
LABELS = CONFIG.get("cls_config", {}).get("labels", []) if CONFIG else []

# mqtt配置
MQTT_BASE_URL = CONFIG.get("mqtt_server", {}).get("base_url", "http://124.71.81.97:18007") if CONFIG else "http://124.71.81.97:18007"
MQTT_KEY = CONFIG.get("mqtt_server", {}).get("key", "ZhuoShang") if CONFIG else "ZhuoShang"



# Sherpa TTS 配置
SHERPA_MODEL_DIR = "/workspace/xiaozhi-esp32-server/main/xiaozhi-server/models/sherpa-tts"
SHERPA_ACOUSTIC_DIRNAME = CONFIG.get("sherpa_tts", {}).get("acoustic_model_dirname", "matcha-icefall-zh-en") if CONFIG else "matcha-icefall-zh-en"
SHERPA_VOCODER_NAME = CONFIG.get("sherpa_tts", {}).get("vocoder_filename", "vocos-16khz-univ.onnx") if CONFIG else "vocos-16khz-univ.onnx"
SHERPA_NUM_THREADS = int(CONFIG.get("sherpa_tts", {}).get("num_threads", 2)) if CONFIG else 2
SHERPA_SPEED = float(CONFIG.get("sherpa_tts", {}).get("speed", 1.0)) if CONFIG else 1.0
SHERPA_SID = int(CONFIG.get("sherpa_tts", {}).get("sid", 0)) if CONFIG else 0
TTS_OUTPUT_DIR = CONFIG.get("tts_output_dir", "/workspace/PaddleSpeech-api/audiofile") if CONFIG else "/workspace/PaddleSpeech-api/audiofile"

# app_server 配置
APP_SERVER_BASE_URL =CONFIG.get("app_server", {}).get("base_url", "http://124.71.81.97:18007") if CONFIG else "http://124.71.81.97:18007"


# 挂载静态文件目录，允许通过 /audiofile/ 路径访问文件
app.mount("/audiofile", StaticFiles(directory=TTS_OUTPUT_DIR), name="audiofile")


def build_sherpa_tts():
    """构建 Sherpa TTS 引擎"""
    acoustic_dir = os.path.join(SHERPA_MODEL_DIR, SHERPA_ACOUSTIC_DIRNAME)
    vocoder_path = os.path.join(SHERPA_MODEL_DIR, SHERPA_VOCODER_NAME)

    rule_fsts = ",".join([
        os.path.join(acoustic_dir, "phone-zh.fst"),
        os.path.join(acoustic_dir, "date-zh.fst"),
        os.path.join(acoustic_dir, "number-zh.fst"),
    ])

    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=os.path.join(acoustic_dir, "model-steps-3.onnx"),
                vocoder=vocoder_path,
                lexicon=os.path.join(acoustic_dir, "lexicon.txt"),
                tokens=os.path.join(acoustic_dir, "tokens.txt"),
                data_dir=os.path.join(acoustic_dir, "espeak-ng-data"),
            ),
            num_threads=SHERPA_NUM_THREADS,
            debug=False,
        ),
        max_num_sentences=1,
        rule_fsts=rule_fsts,
    )

    if not cfg.validate():
        raise ValueError("Sherpa OfflineTtsConfig 校验失败，请检查模型路径与文件完整性")

    return sherpa_onnx.OfflineTts(cfg)


# 全局初始化 Sherpa TTS 引擎
_sherpa_tts = None


def get_sherpa_tts():
    """懒加载 Sherpa TTS 引擎"""
    global _sherpa_tts
    if _sherpa_tts is None:
        _sherpa_tts = build_sherpa_tts()
        logger.info("Sherpa TTS 引擎初始化完成")
    return _sherpa_tts


def check_and_update_cache(client_id: str, text: str) -> bool:
    """
    检查缓存并更新缓存记录
    :param client_id: 客户端ID，作为缓存key
    :param text: 匹配的分类文本
    :return: True表示需要调用send_listen_command，False表示命中缓存且text相同跳过调用
    """
    current_time = time.time()
    with _cache_lock:
        # 检查缓存是否存在且未过期
        if client_id in _command_cache:
            cache_entry = _command_cache[client_id]
            if current_time - cache_entry["timestamp"] < _CACHE_TTL:
                # 缓存命中且未过期，判断text是否相同
                if cache_entry["text"] == text:
                    logger.info(f"client_id={client_id} 命中缓存(text相同)，跳过调用")
                    return False
                else:
                    logger.info(f"client_id={client_id} 缓存存在但text不同({cache_entry['text']} -> {text})，允许调用")
        # 更新缓存
        _command_cache[client_id] = {
            "text": text,
            "timestamp": current_time
        }
        # 清理过期缓存（超过缓存有效期2倍的记录）
        expired_keys = [
            k for k, v in _command_cache.items()
            if current_time - v["timestamp"] >= _CACHE_TTL * 2
        ]
        for k in expired_keys:
            del _command_cache[k]
        return True

class TTSRequest(BaseModel):
    name: str
    client_id: str
    text: str
    output_file: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """验证 text 参数"""
        if not v or not v.strip():
            raise ValueError("text 不能为空")
        if len(v) > 1000:
            raise ValueError("text 长度不能超过 1000 字符")
        return v.strip()

    @field_validator('output_file')
    @classmethod
    def validate_output_file(cls, v: str) -> str:
        """验证 output_file 参数，防止路径遍历"""
        import re
        if not v or not v.strip():
            raise ValueError("output_file 不能为空")
        v = v.strip()
        if os.path.sep in v or (os.path.altsep and os.path.altsep in v):
            raise ValueError("output_file 只能是文件名，不能包含路径")
        if v.startswith('.') and len(v) > 1 and v[1] == '.':
            raise ValueError("output_file 不能包含路径遍历字符")
        filename = os.path.basename(v)
        if filename != v:
            raise ValueError("output_file 只能是文件名，不能包含路径")
        return filename

class AudioDataRequest(BaseModel):
    """音频数据请求模型"""

    client_id: str
    data: str

    @field_validator('client_id')
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        """验证 client_id 参数"""
        if not v or not v.strip():
            raise ValueError("client_id 不能为空")
        return v

    @field_validator('data')
    @classmethod
    def validate_data(cls, v: str) -> str:
        """验证 data 参数"""
        if not v or not v.strip():
            raise ValueError("data 不能为空")
        return v


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "PaddleSpeech 音频分类 API",
        "version": "1.0.0",
        "endpoints": {
            "/upload/audio": "POST - 接收音频 base64 数据并保存为文件"
        }
    }

@app.post("/tts")
async def tts(request: TTSRequest):
    """TTS 语音合成接口，基于 Sherpa-ONNX 实现"""
    try:
        tts_engine = get_sherpa_tts()

        # 清理文本
        text = request.text.replace("~", "").replace("～", "")
        if not text:
            raise HTTPException(status_code=400, detail="文本处理后为空")

        # 生成合成音频
        audio = tts_engine.generate(text, sid=SHERPA_SID, speed=SHERPA_SPEED)

        # 将音频数据转换为 WAV 格式
        samples = audio.samples
        sample_rate = int(audio.sample_rate)
        arr = np.asarray(samples)
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, -1.0, 1.0)
            int_samples = (arr * 32767.0).astype(np.int16)
        else:
            int_samples = arr.astype(np.int16)

        # 保存 WAV 文件
        os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(TTS_OUTPUT_DIR, request.output_file)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int_samples.tobytes())

        wav_data = wav_buffer.getvalue()
        with open(output_path, "wb") as f:
            f.write(wav_data)

        # 计算音频时长（秒）
        duration = len(int_samples) / sample_rate

        logger.info(f"TTS合成完成: {output_path}, 时长: {duration:.2f}s")
      
        try:
            if "@@@" in request.client_id:
                code = request.client_id.split("@@@")[1]
            else:
                code = request.client_id
            
            # 使用f-string拼接URL
            file_url = f"http://{API_IP}:{API_PORT}/{output_path.lstrip('/')}"
            
            logger.info(f"开始发送消息到应用服务器: code={code}, url={file_url}")
            
            http_client = CommandHttpClient(base_url=APP_SERVER_BASE_URL,key="")
            result = http_client.send_message(code, file_url, request.name)  # 需确保name已定义
            
            logger.info(f"返回消息：{result}")

            # 校验发送结果
            if result:
                logger.info(f"消息发送成功")
            else:
                logger.warning(f"消息发送失败,但继续处理")
            
        except Exception as e:
            logger.error(f"发送消息到应用服务器失败: {e}")

        return {
            "success": True,
            "message": "TTS成功",
            "data": {
                "save_path": output_path,
                "duration": round(duration, 2)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS失败: {str(e)}")
        return {
            "success": False,
            "message": f"TTS失败: {type(e).__name__}"
        }


@app.post("/upload/audio")
async def upload_audio(request: AudioDataRequest):
    """
    接收音频 base64 数据并保存为文件

    请求体示例:
    ```json
    {
        "client_id": "client123",
        "data": "base64_encoded_audio_data"
    }
    ```
    """
    try:
        if(not LABELS):
            logger.info(f"labels为空")
            return {
                "success": True,
                "message": "labels为空",
                "data": {}
            }

        # 确保 temp 目录存在
        os.makedirs(TEMP_DIR, exist_ok=True)

        # 解码 base64 数据 (OPUS 编码)
        try:
            audio_data = base64.b64decode(request.data)
            # if(request.format=="opus"):
            #     opus_decoder = OpusDecoder()
            #     pcm_data=opus_decoder.decode(audio_data)
            # else:
            pcm_data=audio_data
       

        except Exception as e:
            logger.error(f"OPUS 解码失败: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"OPUS 解码失败: {str(e)}"
            )

        # 生成文件名
        file_id = datetime.now().strftime("%H%M%S")
        wav_filename = f"{request.client_id}_{file_id}.wav"
        wav_path = os.path.join(TEMP_DIR, wav_filename)

        # 保存 pcm 文件
        # 确保数据长度是偶数（16位音频）
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        # 降噪处理
        pcm_data = reduce_noise(pcm_data, SAMPLE_RATE)
        #logger.info("音频降噪处理完成")

        # 创建WAV文件头
        wav_buffer = io.BytesIO()
        success = False
        try:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(16000)  # 16kHz采样率
                wav_file.writeframes(pcm_data)

            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            with open(wav_path, "wb") as f:
                f.write(wav_data)
            # logger.info(f"音频已保存到: {wav_path}")
            
            clsclient_executor = CLSClientExecutor()
            res = clsclient_executor(
            input=wav_path,
            server_ip=DEFAULT_SERVER_IP,
            port=DEFAULT_PORT,
            topk=DEFAULT_TOPK
            )
            # logger.info(f"音频分类结果: {res.json()}")

            # 处理分类结果
            res_data = res.json()
            if res_data.get('success') and res_data.get('result'):
                results = res_data['result'].get('results', [])
                logger.info(f"音频分类结果: {results}")
                matched_classes = []

                for item in results:
                    prob = item.get('prob', 0)
                    class_name = item.get('class_name', '')
                   
                    if prob > SCORE and class_name in LABELS:
                        arrays=LABELS.split(" ")
                        sub_matches = [label for label in arrays if class_name in label]
                        logger.info(f"sub_matches: {sub_matches}")
                        matche=sub_matches[0]
                        if "-" in matche:
                            matche=matche.split("-")[1]
                        matched_classes.append(matche)

                if matched_classes:
                    text =','.join(matched_classes)
                    text=f"发现异常声音："+text
                    # 检查缓存，5分钟内相同client_id不重复调用
                    if not check_and_update_cache(request.client_id, text):
                        logger.info(f"client_id={request.client_id} 跳过重复调用")
                    else:
                        if("@@@" in request.client_id):
                            code=request.client_id.split("@@@")[1]
                        else:
                            code=request.client_id
                        if "_" in code:
                            code=code.replace("_", ":")
                        # 从配置文件读取 HTTP 客户端参数
                        http_client = CommandHttpClient(base_url=APP_SERVER_BASE_URL,key="")
                        result=http_client.send_alarm(macAddress=code,reminder=text)
                        logger.info(f"返回消息：{result}")
            success = True
        except Exception as e:
            logger.error(f"WAV转换失败: {e}")
        
        # 删除临时文件
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        if(success):
            return {
                "success": True,
                "message": "音频文件处理成功",
                "data": {
                    "client_id": request.client_id
                }
            }
        else:
            return {
                "success": False,
                "message": "音频文件处理失败"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"音频文件保存失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"音频文件保存失败: {str(e)}"
        )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_IP, port=API_PORT)
