"""
异常声音告警确认：待确认会话、话术生成、超时与回调处理。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from threading import Lock, Timer
from typing import Dict, List, Optional, Tuple

from http_client import CommandHttpClient

logger = logging.getLogger(__name__)

_pending_alarms: Dict[str, "PendingAlarm"] = {}
_pending_lock = Lock()
# client_id -> session_id，同一设备同时仅允许一个待确认会话
_client_pending: Dict[str, str] = {}

# 多种声音命中时的优先级（英文 class_name）
_SOUND_PRIORITY = [
    "Distress",
    "Screaming",
    "Squeal",
    "Shout",
    "Yell",
    "Bellow",
    "Crying",
    "Explosion",
    "Boom",
    "Bang",
    "Glass",
    "Breaking",
    "Slam",
    "Chink",
    "Sneeze",
    "Cough",
]

# 中文标签 -> 口语描述
_CN_SOUND_DESC: Dict[str, str] = {
    "求救": "好像有人在呼救",
    "呼喊": "好像有人在呼救",
    "叫喊": "好像有人在呼救",
    "咆哮": "好像有人在呼救",
    "尖叫": "好像有人在尖叫",
    "尖叫声": "好像有人在尖叫",
    "哭泣": "好像有人在哭",
    "玻璃破碎": "好像有东西碎裂了",
    "破碎声": "好像有东西碎裂了",
    "碰撞声": "好像有东西碎裂了",
    "爆炸": "好像有一声巨响",
    "轰鸣": "好像有一声巨响",
    "爆裂声": "好像有一声巨响",
    "猛击": "好像有一声巨响",
    "咳嗽": "好像有人身体不适",
    "打喷嚏": "好像有人身体不适",
}

_DEFAULT_INQUIRY_TEMPLATE = (
    "{sounds_desc}，您还好吗？需要我帮您通知一下家人吗？"
)


@dataclass
class PendingAlarm:
    session_id: str
    client_id: str
    mac_code: str
    reminder: str
    created_at: float
    status: str = "pending"
    timer: Optional[Timer] = field(default=None, repr=False)


@dataclass
class AlarmConfirmConfig:
    enabled: bool = True
    timeout_seconds: int = 30
    inquiry_template: str = _DEFAULT_INQUIRY_TEMPLATE
    confirm_reply: str = "好的，我已经帮您通知家人了。"
    cancel_reply: str = "好的，那我就不打扰了，您有需要随时叫我。"
    fallback_direct_alarm: bool = True


def normalize_label_entries(labels) -> List[str]:
    """将 config 中的 labels（list 或 str）统一为字符串列表。"""
    if isinstance(labels, list):
        return [str(x).strip() for x in labels if x]
    if isinstance(labels, str):
        return [x.strip() for x in labels.split() if x.strip()]
    return []


def match_chinese_label(class_name: str, label_entries: List[str]) -> Optional[str]:
    """根据 PaddleSpeech 英文 class_name 匹配配置标签，返回中文名。"""
    for entry in label_entries:
        if class_name in entry:
            if "-" in entry:
                return entry.split("-", 1)[1]
            return entry
    return None


def pick_sounds_desc(matched_cn_labels: List[str], matched_en_classes: List[str]) -> str:
    """按优先级选取一条口语化描述。"""
    if len(matched_cn_labels) > 1:
        return "好像有些不寻常的声响"

    if matched_cn_labels:
        return _CN_SOUND_DESC.get(matched_cn_labels[0], "好像有些不寻常的声响")

    # 仅有英文分类命中时的兜底
    en_to_desc = {
        "Shout": "好像有人在呼救",
        "Yell": "好像有人在呼救",
        "Bellow": "好像有人在呼救",
        "Screaming": "好像有人在尖叫",
        "Squeal": "好像有人在尖叫",
        "Crying": "好像有人在哭",
        "Glass": "好像有东西碎裂了",
        "Breaking": "好像有东西碎裂了",
        "Chink": "好像有东西碎裂了",
        "Explosion": "好像有一声巨响",
        "Boom": "好像有一声巨响",
        "Bang": "好像有一声巨响",
        "Slam": "好像有一声巨响",
        "Sneeze": "好像有人身体不适",
        "Cough": "好像有人身体不适",
    }
    for en in _SOUND_PRIORITY:
        if en in matched_en_classes:
            return en_to_desc.get(en, "好像有些不寻常的声响")

    return "好像有些不寻常的声响"


def build_inquiry_payload(
    session_id: str,
    api_base_url: str,
    sounds_desc: str,
    template: str,
) -> str:
    """构造带协议前缀的 listen 命令文本（前缀不播报）。"""
    body = template.format(sounds_desc=sounds_desc)
    return f"[ALARM_CONFIRM:{session_id}|{api_base_url}]{body}"


def _cancel_timer(pending: PendingAlarm) -> None:
    if pending.timer is not None:
        pending.timer.cancel()
        pending.timer = None


def _cleanup_session(session_id: str) -> None:
    with _pending_lock:
        pending = _pending_alarms.pop(session_id, None)
        if pending:
            _cancel_timer(pending)
            _client_pending.pop(pending.client_id, None)


def _send_alarm(mac_code: str, reminder: str, app_server_base_url: str) -> dict:
    client = CommandHttpClient(base_url=app_server_base_url, key="")
    return client.send_alarm(macAddress=mac_code, reminder=reminder)


def _on_timeout(
    session_id: str,
    app_server_base_url: str,
) -> None:
    with _pending_lock:
        pending = _pending_alarms.get(session_id)
        if not pending or pending.status != "pending":
            return
        pending.status = "timeout"
        _cancel_timer(pending)

    try:
        result = _send_alarm(pending.mac_code, pending.reminder, app_server_base_url)
        logger.info(f"告警确认超时，已自动发送告警: session={session_id}, result={result}")
    except Exception as e:
        logger.error(f"超时发送告警失败: session={session_id}, error={e}")
    finally:
        _cleanup_session(session_id)


def has_client_pending(client_id: str) -> bool:
    with _pending_lock:
        session_id = _client_pending.get(client_id)
        if not session_id:
            return False
        pending = _pending_alarms.get(session_id)
        return pending is not None and pending.status == "pending"


def start_alarm_confirm(
    *,
    client_id: str,
    mac_code: str,
    reminder: str,
    sounds_desc: str,
    api_base_url: str,
    mqtt_base_url: str,
    mqtt_key: str,
    app_server_base_url: str,
    cfg: AlarmConfirmConfig,
) -> Tuple[bool, Optional[str]]:
    """
    发起告警确认询问。成功返回 (True, session_id)，失败返回 (False, None)。
    """
    if has_client_pending(client_id):
        logger.info(f"client_id={client_id} 已有待确认告警，跳过")
        return False, None

    session_id = uuid.uuid4().hex
    inquiry_text = build_inquiry_payload(
        session_id, api_base_url, sounds_desc, cfg.inquiry_template
    )

    try:
        mqtt_client = CommandHttpClient(base_url=mqtt_base_url, key=mqtt_key)
        mqtt_client.send_listen_command(client_id=client_id, text=inquiry_text)
    except Exception as e:
        logger.error(f"发送告警询问失败: client_id={client_id}, error={e}")
        if cfg.fallback_direct_alarm:
            try:
                _send_alarm(mac_code, reminder, app_server_base_url)
                logger.warning(f"MQTT 询问失败，已降级直接发送告警: client_id={client_id}")
                return True, None
            except Exception as alarm_err:
                logger.error(f"降级发送告警失败: {alarm_err}")
        return False, None

    import time

    pending = PendingAlarm(
        session_id=session_id,
        client_id=client_id,
        mac_code=mac_code,
        reminder=reminder,
        created_at=time.time(),
    )
    timer = Timer(
        cfg.timeout_seconds,
        _on_timeout,
        args=(session_id, app_server_base_url),
    )
    pending.timer = timer

    with _pending_lock:
        _pending_alarms[session_id] = pending
        _client_pending[client_id] = session_id

    timer.start()
    logger.info(
        f"已发起告警确认询问: session={session_id}, client_id={client_id}, "
        f"timeout={cfg.timeout_seconds}s"
    )
    return True, session_id


def handle_alarm_response(
    session_id: str,
    confirmed: bool,
    app_server_base_url: str,
) -> str:
    """
    处理用户确认回调。返回 action: alarm_sent / alarm_cancelled / session_not_found。
    """
    with _pending_lock:
        pending = _pending_alarms.get(session_id)
        if not pending or pending.status != "pending":
            return "session_not_found"
        pending.status = "confirmed" if confirmed else "cancelled"
        _cancel_timer(pending)

    try:
        if confirmed:
            result = _send_alarm(pending.mac_code, pending.reminder, app_server_base_url)
            logger.info(f"用户确认发送告警: session={session_id}, result={result}")
            action = "alarm_sent"
        else:
            logger.info(f"用户取消告警: session={session_id}")
            action = "alarm_cancelled"
    except Exception as e:
        logger.error(f"处理告警确认失败: session={session_id}, error={e}")
        action = "session_not_found" if not confirmed else "alarm_sent"
    finally:
        _cleanup_session(session_id)

    return action


def load_alarm_confirm_config(config: Optional[dict]) -> AlarmConfirmConfig:
    raw = (config or {}).get("alarm_confirm", {}) or {}
    return AlarmConfirmConfig(
        enabled=bool(raw.get("enabled", True)),
        timeout_seconds=int(raw.get("timeout_seconds", 30)),
        inquiry_template=str(
            raw.get("inquiry_template", _DEFAULT_INQUIRY_TEMPLATE)
        ),
        confirm_reply=str(
            raw.get("confirm_reply", "好的，我已经帮您通知家人了。")
        ),
        cancel_reply=str(
            raw.get("cancel_reply", "好的，那我就不打扰了，您有需要随时叫我。")
        ),
        fallback_direct_alarm=bool(raw.get("fallback_direct_alarm", True)),
    )
