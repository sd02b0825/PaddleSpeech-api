"""SenseVoiceSmall 语音识别工具类。"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_asr_instance: Optional["SenseVoiceAsr"] = None
_init_lock = Lock()


class SenseVoiceAsr:
    """基于 FunASR SenseVoiceSmall 的语音识别工具。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        language: str = "auto",
        use_itn: bool = True,
    ):
        if not model_path:
            raise ValueError("model_path 不能为空")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SenseVoice 模型路径不存在: {model_path}")

        from funasr import AutoModel

        self.model_path = model_path
        self.device = device
        self.language = language
        self.use_itn = use_itn
        self._infer_lock = Lock()
        self._model = AutoModel(
            model=model_path,
            device=device,
            disable_update=True,
        )
        logger.info(
            "SenseVoice ASR 初始化完成: model_path=%s, device=%s",
            model_path,
            device,
        )

    def recognize(self, wav_path: str) -> str:
        """
        对 wav 音频文件进行语音识别。

        Args:
            wav_path: wav 音频文件路径

        Returns:
            清洗后的识别文本

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 识别失败或结果为空
        """
        if not wav_path or not os.path.isfile(wav_path):
            raise FileNotFoundError(f"音频文件不存在: {wav_path}")

        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        with self._infer_lock:
            try:
                res = self._model.generate(
                    input=wav_path,
                    cache={},
                    language=self.language,
                    use_itn=self.use_itn,
                )
            except Exception as e:
                raise RuntimeError(f"SenseVoice 识别失败: {e}") from e

        if not res or not isinstance(res, list) or "text" not in res[0]:
            raise RuntimeError(f"SenseVoice 返回结果无效: {res}")

        text = rich_transcription_postprocess(res[0]["text"])
        return text.strip() if isinstance(text, str) else str(text)

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "SenseVoiceAsr":
        """从全局配置字典构建实例。"""
        raw = (config or {}).get("sensevoice_asr", {}) or {}
        model_path = str(raw.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("config.sensevoice_asr.model_path 未配置")

        return cls(
            model_path=model_path,
            device=str(raw.get("device", "cpu")),
            language=str(raw.get("language", "auto")),
            use_itn=bool(raw.get("use_itn", True)),
        )


def get_sensevoice_asr(config: Optional[dict] = None) -> SenseVoiceAsr:
    """线程安全的 SenseVoice ASR 全局单例（双重检查锁）。"""
    global _asr_instance
    if _asr_instance is None:
        with _init_lock:
            if _asr_instance is None:
                _asr_instance = SenseVoiceAsr.from_config(config)
    return _asr_instance
