"""
PaddleSpeech 音频分类 API
基于 FastAPI 实现，提供音频分类检测接口
"""
import asyncio
import io
import wave
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import sherpa_onnx


def _resolve_log_level() -> int:
    """从 config.yaml 读取日志级别，默认 INFO（过滤 DEBUG）。"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        level_name = str((cfg.get("server", {}) or {}).get("log_level", "INFO")).upper()
        return getattr(logging, level_name, logging.INFO)
    except Exception:
        return logging.INFO


def setup_logging() -> None:
    """配置 root logger，使各模块日志统一写入 logs/api.log。"""
    log_level = _resolve_log_level()
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "api.log")
    handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.suffix = "%Y-%m-%d.log"
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(handler)


setup_logging()

from http_client import CommandHttpClient
from alarm_confirm import (
    load_alarm_confirm_config,
    normalize_label_entries,
    handle_alarm_response,
)
from sensevoice_asr import get_sensevoice_asr
from audio_upload import RawAudioUpload, UploadAudioConfig, process_upload_audio

logger = logging.getLogger(__name__)


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
SERVER_URL = CONFIG.get("server", {}).get("url", "http://124.71.81.97:8091") if CONFIG else "http://124.71.81.97:8091"

# 从配置文件读取参数
DEFAULT_SERVER_IP = CONFIG.get("cls_config", {}).get("ip", "127.0.0.1") if CONFIG else "127.0.0.1"
DEFAULT_PORT = CONFIG.get("cls_config", {}).get("port", 8090) if CONFIG else 8090
DEFAULT_TOPK = CONFIG.get("cls_config", {}).get("topk", 1) if CONFIG else 1
TEMP_DIR = CONFIG.get("cls_config", {}).get("temp_dir", "/workspace/temp") if CONFIG else "/workspace/temp"
SAMPLE_RATE = CONFIG.get("cls_config", {}).get("sample_rate", 16000) if CONFIG else 16000
SCORE = CONFIG.get("cls_config", {}).get("score", 0.4) if CONFIG else 0.4
LABEL_ENTRIES = normalize_label_entries(
    CONFIG.get("cls_config", {}).get("labels", []) if CONFIG else []
)
ALARM_CONFIRM_CFG = load_alarm_confirm_config(CONFIG)

_SPEECH_DETECT = (CONFIG.get("speech_detect", {}) if CONFIG else {}) or {}
SPEECH_DETECT_ENABLED = bool(_SPEECH_DETECT.get("enabled", True))
SPEECH_LABELS = normalize_label_entries(_SPEECH_DETECT.get("speech_labels", []))
DISTRESS_KEYWORDS = normalize_label_entries(
    _SPEECH_DETECT.get("distress_keywords", [])
)
DISTRESS_LABEL = str(_SPEECH_DETECT.get("distress_label", "求救")).strip() or "求救"

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
APP_SERVER_BASE_URL = CONFIG.get("app_server", {}).get("base_url", "http://124.71.81.97:18007") if CONFIG else "http://124.71.81.97:18007"

UPLOAD_CFG = UploadAudioConfig(
    cls_server_ip=DEFAULT_SERVER_IP,
    cls_port=DEFAULT_PORT,
    cls_topk=DEFAULT_TOPK,
    temp_dir=TEMP_DIR,
    sample_rate=SAMPLE_RATE,
    score=SCORE,
    label_entries=LABEL_ENTRIES,
    speech_detect_enabled=SPEECH_DETECT_ENABLED,
    speech_labels=SPEECH_LABELS,
    distress_keywords=DISTRESS_KEYWORDS,
    distress_label=DISTRESS_LABEL,
    sensevoice_config=CONFIG,
    mqtt_base_url=MQTT_BASE_URL,
    mqtt_key=MQTT_KEY,
    app_server_base_url=APP_SERVER_BASE_URL,
    server_url=SERVER_URL,
    alarm_confirm_cfg=ALARM_CONFIRM_CFG,
)

# 挂载静态文件目录，允许通过 /audiofile/ 路径访问文件
app.mount("/audiofile", StaticFiles(directory=TTS_OUTPUT_DIR), name="audiofile")


@app.on_event("startup")
async def preload_sensevoice_asr():
    """启动时预加载 SenseVoice，失败则降级（说话声 ASR 跳过）。"""
    if not SPEECH_DETECT_ENABLED:
        logger.info("说话声检测已关闭，跳过 SenseVoice ASR 预加载")
        return
    try:
        get_sensevoice_asr(CONFIG)
        logger.info("SenseVoice ASR 启动预加载完成")
    except Exception as e:
        logger.error(f"SenseVoice ASR 启动预加载失败，说话声求救检测将降级: {e}")


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


class TTSRequest(BaseModel):
    receiver: str
    speaker: str
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


class AlarmResponseRequest(BaseModel):
    """告警确认回调请求"""

    session_id: str
    confirmed: bool
    text: str = ""

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("session_id 不能为空")
        return v.strip()


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "PaddleSpeech 音频分类 API",
        "version": "1.0.0",
        "endpoints": {
            "/upload/audio": "POST - 接收硬件原始 PCM（binary body + headers）",
            "/alarm/response": "POST - 接收告警确认结果",
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

            file_url = f"{SERVER_URL}/audiofile/{request.output_file}"

            logger.info(f"开始发送消息到应用服务器: code={code}, url={file_url}")

            http_client = CommandHttpClient(base_url=APP_SERVER_BASE_URL, key="")
            result = http_client.send_message(code, file_url, request.receiver, request.speaker)

            logger.info(f"返回消息：{result}")

            if isinstance(result, dict) and result.get('code') == 200:
                logger.info(f"消息发送成功")
            else:
                logger.warning(f"消息发送失败:{result}")
                return {
                    "success": False,
                    "message": f"消息发送失败:{result}"
                }

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


@app.post("/alarm/response")
async def alarm_response(request: AlarmResponseRequest):
    """接收 xiaozhi-server 转发的用户告警确认结果"""
    action = handle_alarm_response(
        session_id=request.session_id,
        confirmed=request.confirmed,
        app_server_base_url=APP_SERVER_BASE_URL,
    )
    if action == "session_not_found":
        return {
            "success": False,
            "message": "会话不存在或已过期",
            "data": {"session_id": request.session_id, "action": action},
        }
    message = "告警已发送" if action == "alarm_sent" else "告警已取消"
    return {
        "success": True,
        "message": message,
        "data": {"session_id": request.session_id, "action": action},
    }


@app.post("/upload/audio")
async def upload_audio(request: Request):
    """
    接收硬件上传的原始 PCM 音频数据。

    硬件协议（与 AudioMonitor::UploadAudio 对齐）：
        HTTP POST，body 为原始 PCM 二进制（application/octet-stream），
        元数据通过 HTTP header 传递：
            Client-Id        客户端 ID（必填）
            Device-Id        设备 MAC 地址（可选）
            X-Sample-Rate    采样率（Hz，缺省取服务端配置）
            X-Channels       声道数（必须为 1）
            X-Bits           采样位深（必须为 16）
    """
    client_id = ""
    try:
        client_id = (request.headers.get("Client-Id") or "").strip()
        if not client_id:
            logger.warning("音频上传拒绝: Client-Id header 缺失")
            raise HTTPException(status_code=400, detail="Client-Id header 缺失")

        device_id = (request.headers.get("Device-Id") or "").strip()

        def _parse_int_header(name: str, default: int) -> int:
            raw = request.headers.get(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning(
                    f"音频上传拒绝: client_id={client_id}, 无效的 {name} header: {raw!r}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的 {name} header: {raw!r}",
                )

        sample_rate = _parse_int_header("X-Sample-Rate", SAMPLE_RATE)
        channels = _parse_int_header("X-Channels", 1)
        bits = _parse_int_header("X-Bits", 16)

        pcm_data = await request.body()
        if not pcm_data:
            logger.warning(f"音频上传拒绝: client_id={client_id}, 请求体为空")
            raise HTTPException(status_code=400, detail="请求体为空")

        duration_sec = len(pcm_data) / (sample_rate * channels * (bits // 8))
        # logger.info(
        #     f"收到音频上传: client_id={client_id}, device_id={device_id or '-'}, "
        #     f"pcm={len(pcm_data)} bytes, 约{duration_sec:.2f}s, "
        #     f"sample_rate={sample_rate}, channels={channels}, bits={bits}"
        # )

        upload = RawAudioUpload(
            client_id=client_id,
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            channels=channels,
            bits=bits,
            device_id=device_id,
        )

        result = await asyncio.to_thread(process_upload_audio, upload, UPLOAD_CFG)

        if result.get("success"):
            data = result.get("data") or {}
            if data.get("skipped"):
                logger.info(
                    f"音频上传已跳过: client_id={client_id}, "
                    f"reason={data.get('reason', '-')}, message={result.get('message')}"
                )
            else:
                cls_results = data.get("result") or []
                logger.info(
                    f"音频上传处理成功: client_id={client_id}, "
                    f"分类条数={len(cls_results)}, message={result.get('message')}"
                )
        else:
            logger.warning(
                f"音频上传处理失败: client_id={client_id}, message={result.get('message')}"
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"音频上传异常: client_id={client_id or '-'}, error={e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"音频文件保存失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    _uvicorn_log_level = logging.getLevelName(_resolve_log_level()).lower()
    uvicorn.run(
        app,
        host=SERVER_IP,
        port=SERVER_PORT,
        log_level=_uvicorn_log_level,
    )
