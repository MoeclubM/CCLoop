"""
JSON 解析增强模块
"""

import json
import re
from typing import Any, Dict, List, Optional


def clean_invalid_unicode(text: str) -> str:
    """移除无效的 Unicode 代理字符"""
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        return re.sub(r'[\ud800-\udfff]', '', text)


def detect_and_decode(data: bytes) -> str:
    """自动检测编码并解码字节数据"""
    if data.startswith(b'\xff\xfe'):
        return data[2:].decode('utf-16-le')
    if data.startswith(b'\xfe\xff'):
        return data[2:].decode('utf-16-be')
    if data.startswith(b'\xef\xbb\xbf'):
        return data[3:].decode('utf-8')

    encodings = ['utf-8', 'utf-8-sig', 'utf-16-le', 'utf-16-be', 'gbk', 'latin1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except Exception:
            continue

    return data.decode('utf-8', errors='replace')


def extract_first_json(text: str) -> Dict[str, Any]:
    """从文本中提取第一个 JSON 对象（增强版）"""
    text = clean_invalid_unicode(text).strip()

    # 1. 尝试直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2. 尝试用正则提取 JSON 对象
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"done": False, "summary": "Judge解析失败", "next_prompt": "继续尝试"}

    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("解析结果不是 JSON 对象")
        return obj
    except json.JSONDecodeError:
        pass

    # 3. 尝试修复常见的 JSON 格式问题
    try:
        fixed = _fix_json_text(m.group(0))
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {"done": False, "summary": "Judge JSON无效", "next_prompt": "继续"}


def _fix_json_text(text: str) -> str:
    """尝试修复常见的 JSON 格式问题"""
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    text = re.sub(r'[^}{]*$', '', text).strip()
    if text and text[-1] not in '}]':
        last_brace = max(text.rfind('}'), text.rfind(']'))
        if last_brace > 0:
            text = text[:last_brace + 1]
    return text


class JSONBuffer:
    """流式 JSON 缓冲器，用于处理不完整的 JSON 块"""

    def __init__(self):
        self.buffer = ""
        self.decoder = json.JSONDecoder()

    def feed(self, data: str) -> List[Dict[str, Any]]:
        """向缓冲器添加数据，返回解析出的完整 JSON 对象"""
        self.buffer += data
        objects = []

        while True:
            self.buffer = self.buffer.lstrip("\r\n\t ")
            if not self.buffer:
                break

            try:
                obj, idx = self.decoder.raw_decode(self.buffer)
                self.buffer = self.buffer[idx:]
                if isinstance(obj, dict):
                    objects.append(obj)
                elif isinstance(obj, list):
                    objects.extend(item for item in obj if isinstance(item, dict))
            except json.JSONDecodeError:
                break
            except Exception:
                self.buffer = ""
                break

        return objects

    def clear(self) -> None:
        """清空缓冲器"""
        self.buffer = ""

    def has_data(self) -> bool:
        """检查是否有未完成的缓冲数据"""
        return bool(self.buffer.strip())
