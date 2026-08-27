"""
/upload/audio 音频上传分类与告警触发。
"""
from __future__ import annotations

import io
import logging
import os
import re
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from paddlespeech.server.bin.paddlespeech_client import CLSClientExecutor

from alarm_confirm import (
    AlarmConfirmConfig,
    has_client_pending,
    match_chinese_label,
    pick_sounds_desc,
    start_alarm_confirm,
)
from audio_util import reduce_noise
from http_client import CommandHttpClient
from sensevoice_asr import get_sensevoice_asr

logger = logging.getLogger(__name__)

# 告警去重缓存: {client_id: {"text": matched_text, "timestamp": float}}
_command_cache: Dict[str, dict] = {}
_cache_lock = Lock()
_CACHE_TTL = 180

# 上一片 PCM 缓存，用于与当前片拼接成约 6 秒再检测
# 结构: {client_id: {"pcm": bytes, "timestamp": float}}
_pcm_cache: Dict[str, dict] = {}
_pcm_cache_lock = Lock()
_PCM_CACHE_TTL = 10

# 临时目录初始化保护，避免每次请求都调用 makedirs
_temp_dir_ready = False
_temp_dir_lock = Lock()

# CLS 客户端复用单例，避免每次请求重新初始化
_cls_executor: Optional[CLSClientExecutor] = None
_cls_executor_lock = Lock()


def _get_cls_executor() -> CLSClientExecutor:
    """复用 CLSClientExecutor 单例。"""
    global _cls_executor
    if _cls_executor is None:
        with _cls_executor_lock:
            if _cls_executor is None:
                _cls_executor = CLSClientExecutor()
    return _cls_executor


def _ensure_temp_dir(temp_dir: str) -> None:
    """仅在首次调用时创建临时目录。"""
    global _temp_dir_ready
    if _temp_dir_ready:
        return
    with _temp_dir_lock:
        if not _temp_dir_ready:
            os.makedirs(temp_dir, exist_ok=True)
            _temp_dir_ready = True


@dataclass
class UploadAudioConfig:
    """上传音频处理所需配置（由 main 组装注入，避免循环依赖）。"""

    cls_server_ip: str
    cls_port: int
    cls_topk: int
    temp_dir: str
    sample_rate: int
    score: float
    label_entries: List[str]
    speech_detect_enabled: bool
    speech_labels: List[str]
    distress_keywords: List[str]
    distress_label: str
    sensevoice_config: Optional[dict]
    mqtt_base_url: str
    mqtt_key: str
    app_server_base_url: str
    server_url: str
    alarm_confirm_cfg: AlarmConfirmConfig


@dataclass
class RawAudioUpload:
    """硬件上传的原始 PCM 音频请求。

    硬件协议：HTTP POST，body 为原始 PCM 二进制（application/octet-stream），
    元数据通过 HTTP header 传递：
        Client-Id        客户端 ID（必填）
        Device-Id        设备 MAC 地址（可选，用于日志）
        X-Sample-Rate    采样率（Hz）
        X-Channels       声道数（必须为 1）
        X-Bits           采样位深（必须为 16）
    """

    client_id: str
    pcm_data: bytes
    sample_rate: int
    channels: int
    bits: int
    device_id: str = ""


def is_speech_label(class_name: str, speech_labels: list) -> bool:
    """判断分类结果是否属于说话声标签（子串匹配）。"""
    if not class_name or not speech_labels:
        return False
    return any(label == class_name for label in speech_labels)


def contains_distress(text: str, keywords: list) -> bool:
    """识别文本是否包含求救关键词（忽略空白）。"""
    if not text or not keywords:
        return False
    compact = "".join(text.split())
    for kw in keywords:
        if not kw:
            continue
        kw_compact = "".join(kw.split())
        if kw_compact and kw_compact in compact:
            return True
    return False


def check_and_update_cache(client_id: str, text: str) -> bool:
    """
    检查缓存并更新缓存记录。
    :return: True 表示需要触发告警，False 表示命中缓存且 text 相同应跳过
    """
    current_time = time.time()
    with _cache_lock:
        if client_id in _command_cache:
            cache_entry = _command_cache[client_id]
            if current_time - cache_entry["timestamp"] < _CACHE_TTL:
                if cache_entry["text"] == text:
                    logger.info(f"client_id={client_id} 命中缓存(text相同)，跳过调用")
                    return False
                logger.info(
                    f"client_id={client_id} 缓存存在但text不同"
                    f"({cache_entry['text']} -> {text})，允许调用"
                )
        _command_cache[client_id] = {"text": text, "timestamp": current_time}
        expired_keys = [
            k
            for k, v in _command_cache.items()
            if current_time - v["timestamp"] >= _CACHE_TTL * 2
        ]
        for k in expired_keys:
            del _command_cache[k]
        return True


def take_previous_and_store_current(client_id: str, pcm: bytes) -> Optional[bytes]:
    """
    原子认领上一片 PCM，并用当前片覆盖缓存。
    必须在长推理之前调用，缩短同 client 重叠请求的竞态窗口。
    """
    current_time = time.time()
    with _pcm_cache_lock:
        prev_pcm = None
        entry = _pcm_cache.get(client_id)
        if entry:
            if current_time - entry["timestamp"] > _PCM_CACHE_TTL:
                del _pcm_cache[client_id]
            else:
                prev_pcm = entry["pcm"]

        _pcm_cache[client_id] = {"pcm": pcm, "timestamp": current_time}
        expired_keys = [
            k
            for k, v in _pcm_cache.items()
            if current_time - v["timestamp"] >= _PCM_CACHE_TTL * 2
        ]
        for k in expired_keys:
            del _pcm_cache[k]
        return prev_pcm


def _client_mac_code(client_id: str) -> str:
    if "@@@" in client_id:
        code = client_id.split("@@@")[1]
    else:
        code = client_id
    if "_" in code:
        code = code.replace("_", ":")
    return code


def _maybe_trigger_alarm(
    *,
    client_id: str,
    matched_classes: List[str],
    matched_en_classes: List[str],
    cfg: UploadAudioConfig,
) -> None:
    reminder = "发现异常声音：" + ",".join(matched_classes)
    if not check_and_update_cache(client_id, reminder):
        logger.info(f"client_id={client_id} 跳过重复调用")
        return

    code = _client_mac_code(client_id)
    if cfg.alarm_confirm_cfg.enabled:
        sounds_desc = pick_sounds_desc(matched_classes, matched_en_classes)
        ok, session_id = start_alarm_confirm(
            client_id=client_id,
            mac_code=code,
            reminder=reminder,
            sounds_desc=sounds_desc,
            api_base_url=cfg.server_url.rstrip("/"),
            mqtt_base_url=cfg.mqtt_base_url,
            mqtt_key=cfg.mqtt_key,
            app_server_base_url=cfg.app_server_base_url,
            cfg=cfg.alarm_confirm_cfg,
        )
        if not ok:
            logger.info(f"client_id={client_id} 告警确认跳过或失败")
        else:
            logger.info(f"告警确认流程: ok={ok}, session_id={session_id}")
    else:
        http_client = CommandHttpClient(base_url=cfg.app_server_base_url, key="")
        result = http_client.send_alarm(macAddress=code, reminder=reminder)
        logger.info(f"直接发送告警：{result}")


def process_upload_audio(request: RawAudioUpload, cfg: UploadAudioConfig) -> dict:
    """
    同步处理上传音频：降噪、拼接、分类、可选 ASR、告警。
    供 asyncio.to_thread 在线程池中调用。

    硬件上传协议：body 为原始 PCM 二进制，元数据通过 HTTP header 传递，
    已由 main.py 解析为 RawAudioUpload。此处不再做 base64 解码。
    """
    if not cfg.label_entries:
        logger.info("labels为空")
        return {"success": True, "message": "labels为空", "data": {}}

    _ensure_temp_dir(cfg.temp_dir)

    # 协议校验：当前仅支持单声道 16bit PCM
    if request.channels != 1:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持单声道 PCM，收到 channels={request.channels}",
        )
    if request.bits != 16:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 16bit PCM，收到 bits={request.bits}",
        )

    # 采样率优先使用硬件 header 声明的值，缺省回退到服务端配置
    sample_rate = request.sample_rate or cfg.sample_rate
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail="无效的 sample_rate")

    pcm_data = request.pcm_data or b""
    if len(pcm_data) < 2:
        raise HTTPException(status_code=400, detail="音频数据过短")

    if request.device_id:
        logger.info(
            f"client_id={request.client_id} device_id={request.device_id} "
            f"收到 PCM: {len(pcm_data)} bytes, sample_rate={sample_rate}, "
            f"channels={request.channels}, bits={request.bits}"
        )

    # 清洗 client_id 防止 Windows 非法字符（如 : @），并用 uuid 避免同 client 同秒请求互相覆盖
    safe_client = re.sub(r"[^\w.-]", "_", request.client_id)
    file_id = datetime.now().strftime("%H%M%S") + "_" + uuid.uuid4().hex[:8]
    wav_filename = f"{safe_client}_{file_id}.wav"
    wav_path = os.path.join(cfg.temp_dir, wav_filename)

    if len(pcm_data) % 2 != 0:
        pcm_data = pcm_data[:-1]

    pcm_data = reduce_noise(pcm_data, sample_rate)

    # 降噪后立刻原子认领上一片并写入当前片，再进入长推理
    prev_pcm = take_previous_and_store_current(request.client_id, pcm_data)

    # 告警确认期间：同 client_id 直接跳过 CLS/ASR/告警，仅保留 PCM 缓存
    if has_client_pending(request.client_id):
        logger.info(
            f"client_id={request.client_id} 告警确认中，跳过分类与告警处理"
        )
        return {
            "success": True,
            "message": "告警确认中，跳过处理",
            "data": {
                "client_id": request.client_id,
                "skipped": True,
                "reason": "alarm_confirm_pending",
            },
        }

    if prev_pcm:
        detect_pcm = prev_pcm + pcm_data
        # logger.info(
        #     f"client_id={request.client_id} 拼接上一片音频检测: "
        #     f"prev={len(prev_pcm)} + curr={len(pcm_data)} = {len(detect_pcm)} bytes"
        # )
    else:
        detect_pcm = pcm_data
        # logger.info(
        #     f"client_id={request.client_id} 无上一片缓存，单片检测: {len(detect_pcm)} bytes"
        # )

    success = False
    results: List[Any] = []
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(detect_pcm)

        wav_buffer.seek(0)
        wav_data = wav_buffer.read()
        with open(wav_path, "wb") as f:
            f.write(wav_data)

        clsclient_executor = _get_cls_executor()
        res = clsclient_executor(
            input=wav_path,
            server_ip=cfg.cls_server_ip,
            port=cfg.cls_port,
            topk=cfg.cls_topk,
        )

        try:
            res_data = res.json()
        except Exception as e:
            logger.error(
                f"CLS 返回非 JSON 或解析失败: {e}, raw={getattr(res, 'text', res)!r}"
            )
            res_data = {}

        if res_data.get("success") and res_data.get("result"):
            results = res_data["result"].get("results", [])
            logger.info(f"音频分类结果: {results}")
            matched_classes: List[str] = []
            matched_en_classes: List[str] = []

            for item in results:
                prob = item.get("prob", 0)
                class_name = item.get("class_name", "")
                cn_label = match_chinese_label(class_name, cfg.label_entries)
                if prob > cfg.score and cn_label:
                    matched_classes.append(cn_label)
                    matched_en_classes.append(class_name)

            need_asr = (
                cfg.speech_detect_enabled
                and not matched_classes
                and any(
                    item.get("prob", 0) > cfg.score
                    and is_speech_label(item.get("class_name", ""), cfg.speech_labels)
                    and not match_chinese_label(
                        item.get("class_name", ""), cfg.label_entries
                    )
                    for item in results
                )
            )
            if need_asr:
                try:
                    asr = get_sensevoice_asr(cfg.sensevoice_config)
                    asr_text = asr.recognize(wav_path)
                    logger.info(
                        f"说话声 ASR 结果: client_id={request.client_id}, text={asr_text}"
                    )
                    if contains_distress(asr_text, cfg.distress_keywords):
                        if cfg.distress_label not in matched_classes:
                            matched_classes.append(cfg.distress_label)
                            matched_en_classes.append("Distress")
                except Exception as e:
                    logger.error(f"说话声 ASR 失败，跳过求救检测: {e}")

            if matched_classes:
                _maybe_trigger_alarm(
                    client_id=request.client_id,
                    matched_classes=matched_classes,
                    matched_en_classes=matched_en_classes,
                    cfg=cfg,
                )
            success = True
        else:
            logger.warning(f"CLS 服务返回失败: {res_data}")
    except Exception as e:
        logger.error(f"音频处理失败: {e}", exc_info=True)
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError as e:
                logger.warning(f"删除临时 WAV 文件失败: {wav_path}, {e}")

    if success:
        return {
            "success": True,
            "message": "音频文件处理成功",
            "data": {"client_id": request.client_id, "result": results},
        }
    return {"success": False, "message": "音频文件处理失败"}
