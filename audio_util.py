import logging

import noisereduce as nr
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def reduce_noise(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    对 PCM 音频数据进行降噪处理，过滤电流声并突出瞬态变化
    
    Args:
        pcm_data: PCM 原始音频数据（16位小端）
        sample_rate: 采样率，默认 16000
    
    Returns:
        降噪后的 PCM 数据
    """
    try:
        # 将字节数据转换为 numpy 数组（16位整数）
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # 转换为浮点数进行处理（归一化到 [-1, 1]）
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # 1. 高通滤波 - 过滤低频电流声（50Hz以下）
        # 设计高通滤波器，截止频率80Hz，过滤工频干扰和低频电流声
        nyquist = sample_rate / 2
        cutoff_freq = 80  # 截止频率 80Hz
        normalized_cutoff = cutoff_freq / nyquist
        # 确保截止频率有效
        if normalized_cutoff < 1:
            b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
            audio_filtered = signal.filtfilt(b, a, audio_float)
        else:
            audio_filtered = audio_float
        
        # 2. 执行降噪处理（处理非平稳噪声）
        reduced_noise = nr.reduce_noise(
            y=audio_filtered,
            sr=sample_rate,
            prop_decrease=0.8,  # 降噪强度
            stationary=False    # 非平稳噪声
        )
        
        # 3. 瞬态增强 - 突出瞬态变化的声音
        # 使用差分增强方法
        # 计算信号的瞬时能量包络
        analytic_signal = signal.hilbert(reduced_noise)
        envelope = np.abs(analytic_signal)
        
        # 平滑包络
        window_size = int(sample_rate * 0.01)  # 10ms窗口
        if window_size < 3:
            window_size = 3
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # 计算包络的变化率（瞬态特征）
        envelope_diff = np.diff(envelope_smooth, prepend=envelope_smooth[0])
        
        # 归一化瞬态增益
        max_diff = np.max(np.abs(envelope_diff))
        if max_diff > 0:
            transient_gain = 1 + 0.3 * (envelope_diff / max_diff)  # 增强系数0.3
        else:
            transient_gain = np.ones_like(envelope_diff)
        
        # 应用瞬态增强
        enhanced_audio = reduced_noise * transient_gain
        
        # 归一化防止削波
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0.95:
            enhanced_audio = enhanced_audio * 0.95 / max_val
        
        # 转回 16 位整数
        reduced_audio = (enhanced_audio * 32768.0).astype(np.int16)
        
        # 防止溢出
        reduced_audio = np.clip(reduced_audio, -32768, 32767)
        
        return reduced_audio.tobytes()
    except Exception as e:
        logger.warning(f"降噪处理失败，返回原始数据: {e}")
        return pcm_data

