try:
    import opuslib
    HAS_OPUS = True
except ImportError:
    HAS_OPUS = False


class OpusFrameDecoder:
    """
    Opus 逐帧解码器（参考 audio_monitor_server.py 的实现）。
    
    设备端 EncodeOpus() 将多个 Opus 帧拼接发送。
    每个帧前有 4 字节小端序 uint32 表示帧长度，
    格式为: [frame_len_4bytes][frame_data][frame_len_4bytes][frame_data]...
    """

    # 音频参数（与设备端 AudioMonitor 一致）
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

    # Opus 帧参数（与设备端一致）
    OPUS_FRAME_DURATION_MS = 60
    OPUS_FRAME_SIZE = SAMPLE_RATE * OPUS_FRAME_DURATION_MS // 1000  # 960 samples

    # 帧长度前缀大小（4 字节小端序 uint32）
    FRAME_LENGTH_PREFIX_SIZE = 4

    def __init__(self):
        if not HAS_OPUS:
            raise RuntimeError("opuslib 未安装，无法解码 Opus 格式")
        self.decoder = opuslib.Decoder(self.SAMPLE_RATE, self.CHANNELS)

    def decode(self, opus_data: bytes) -> bytes | None:
        """
        解码带帧长度前缀的 Opus 数据，返回 PCM bytes。

        Args:
            opus_data: Opus 编码的字节数据，每帧前有 4 字节长度前缀

        Returns:
            解码后的 PCM 数据（16-bit, 16kHz, 单声道），解码失败返回 None
        """
        if not opus_data or not isinstance(opus_data, bytes):
            print("[ERROR] Opus 数据无效或为空")
            return None

        import struct
        pcm_parts = []
        offset = 0
        frame_count = 0
        total_samples = 0

        while offset < len(opus_data):
            # 读取 4 字节帧长度前缀
            if offset + self.FRAME_LENGTH_PREFIX_SIZE > len(opus_data):
                print(f"[WARN] 剩余数据不足帧长度前缀 ({len(opus_data) - offset} bytes), 停止解码")
                break

            frame_len = struct.unpack_from('<I', opus_data, offset)[0]
            offset += self.FRAME_LENGTH_PREFIX_SIZE

            # 校验帧长度合理性
            if frame_len == 0 or offset + frame_len > len(opus_data):
                print(f"[WARN] 无效帧长度 {frame_len} (剩余 {len(opus_data) - offset} bytes), 停止解码")
                break

            # 提取单帧数据
            frame_data = opus_data[offset:offset + frame_len]
            offset += frame_len

            # 解码单帧
            try:
                pcm_frame = self.decoder.decode(frame_data, self.OPUS_FRAME_SIZE)
                pcm_parts.append(pcm_frame)
                frame_count += 1
                total_samples += len(pcm_frame) // self.SAMPLE_WIDTH
            except opuslib.OpusError as e:
                print(f"[WARN] 第 {frame_count + 1} 帧解码失败: {e}, 跳过")
                # 重置解码器状态，避免后续帧也受影响
                self.decoder = opuslib.Decoder(self.SAMPLE_RATE, self.CHANNELS)
                continue

        print(f"[OPUS] 解码完成: {frame_count} 帧, {total_samples} samples, "
              f"{total_samples / self.SAMPLE_RATE:.2f}s")
        return b''.join(pcm_parts)


# 兼容旧类名
OpusDecoder = OpusFrameDecoder