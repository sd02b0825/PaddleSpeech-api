"""
HTTP 客户端类
用于调用外部 API 接口
"""
import hashlib
import logging
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class CommandHttpClient:
    """命令 HTTP 客户端类"""

    def __init__(self, base_url: str, key: str):
        """
        初始化客户端

        Args:
            base_url: API 基础地址
            key: Token 生成密钥
        """
        self.base_url = base_url.rstrip("/")
        self.key = key

    def _generate_token(self) -> str:
        """
        生成当日有效的 Bearer 令牌

        生成规则：
        1. 获取当前日期，格式为 yyyy-MM-dd
        2. 将日期字符串与 Key 连接（格式：日期+Key）
        3. 对连接后的字符串进行 SHA256 哈希计算

        Returns:
            SHA256 哈希字符串
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        combined_str = f"{current_date}{self.key}"
        token = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
        return token

    def send_listen_command(
        self,
        client_id: str,
        text: str,
        command_type: str = "listen",
        state: str = "detect",
        timeout: Optional[float] = None
    ) -> dict:
        """
        发送监听命令

        Args:
            client_id: 客户端 ID
            text: 文本内容
            command_type: 命令类型，默认 "listen"
            state: 状态，默认 "detect"
            timeout: 请求超时时间（秒）

        Returns:
            API 响应数据

        Raises:
            requests.exceptions.RequestException: HTTP 请求失败
        """
        token = self._generate_token()

        url = f"{self.base_url}/api/commands/{client_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = {
            "type": command_type,
            "state": state,
            "text": text
        }

        logger.info(f"发送命令到 {url}, text={text}")

        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP 请求失败: {e}")
            raise

    def send_alarm(
        self,
        macAddress: str,
        reminder: str,
        timeout: Optional[float] = None
    ) -> dict:
        """
        发送告警信息

        Args:
            macAddress: mac地址
            reminder: 提醒内容
            timeout: 请求超时时间（秒）

        Returns:
            API 响应数据，包含 code、message、data 字段

        Raises:
            requests.exceptions.RequestException: HTTP 请求失败
        """
        token = self._generate_token()

        url = f"{self.base_url}/api/app/userInfo/alarm"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "key":"_1622a278e2322d2b21c47c007fa9d131"
        }
        payload = {
            "macAddress": macAddress,
            "reminder": reminder
        }

        logger.info(f"发送告警信息到 {url}, macAddress={macAddress}, reminder={reminder}")

        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP 请求失败: {e}")
            raise

    def send_message(
        self,
        macAddress: str,
        file_url: str,
        name: str,
        timeout: Optional[float] = None
    ) -> dict:
        """
        发送留言信息

        Args:
            macAddress: mac地址
            file_url: 留言文件路径
            name: 声纹称呼
            timeout: 请求超时时间（秒）

        Returns:
            API 响应数据，包含 code、message、data 字段

        Raises:
            requests.exceptions.RequestException: HTTP 请求失败
        """
        token = self._generate_token()

        url = f"{self.base_url}/api/app/userInfo/message"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "key":"_1622a278e2322d2b21c47c007fa9d131"
        }
        payload = {
            "macAddress": macAddress,
            "fileUrl": file_url,
            "name": name
        }

        logger.info(f"发送留言信息到 {url}, macAddress={macAddress}, fileUrl={file_url}, name={name}")

        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP 请求失败: {e}")

