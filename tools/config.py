#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""集中读取 .env 配置，并提供带校验的配置对象。

为什么需要这个模块
------------------
不要在业务代码里散落 ``os.environ["LLM_API_KEY"]`` 这种写法，原因有三：

1. 密钥从哪来、有哪些必填项，会分散到各处，难以维护；
2. 配置缺失往往等到**真正调用 API 时**才报 KeyError，排查成本高；
   这里改为启动时一次性校验，失败即给出可操作的修复指引；
3. 密钥容易被误打印进日志。本模块提供 ``masked()``，只输出打码后的视图。

用法::

    from tools.config import get_config

    cfg = get_config()
    print(cfg.llm_model)   # 直接用属性，无需再判空

自检::

    python tools/config.py     # 打印当前配置（密钥自动打码）
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 本文件位于 <项目根>/tools/config.py，因此上溯两层即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"

# 必填项：缺失则直接拒绝启动
_REQUIRED_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")

# 选填项及其默认值
_DEFAULT_DB_PATH = "法规知识库/law.db"

# 模板里那串占位符 sk-xxxxx…，用来提醒"你还没填真实密钥"
_PLACEHOLDER_KEY = re.compile(r"^sk-x+$", re.IGNORECASE)


class ConfigError(RuntimeError):
    """配置缺失或非法。消息中直接写明修复方法，方便直接照做。"""


@dataclass(frozen=True)
class Config:
    """不可变配置对象。字段用属性访问，拼错名字会立刻报错。"""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    db_path: Path

    def masked(self) -> dict:
        """返回可安全打印 / 写日志的配置视图，密钥只保留首尾各 4 位。"""
        return {
            "LLM_API_KEY": _mask_secret(self.llm_api_key),
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL": self.llm_model,
            "DB_PATH": str(self.db_path),
        }


def _mask_secret(secret: str) -> str:
    """把密钥打码成 ``sk-a****z`` 形式，避免截图/日志泄露。"""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * 8}{secret[-4:]}"


def _build_config() -> Config:
    """读取 .env（若存在）并校验，返回 Config。"""
    # 说明：真实环境变量优先级高于 .env（override=False），
    # 这样 CI / 容器里可以用环境变量临时覆盖，无需改文件。
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)

    missing = [key for key in _REQUIRED_KEYS if not os.environ.get(key, "").strip()]
    if missing:
        if ENV_FILE.exists():
            hint = f"请补全 {ENV_FILE} 中的以下键："
        else:
            hint = (
                f"尚未创建配置文件 {ENV_FILE}，请先复制模板：\n"
                f"    Copy-Item .env.example .env      # PowerShell\n"
                f"    cp .env.example .env             # bash / WSL\n"
                f"然后填入真实值，需要包含以下键："
            )
        raise ConfigError(f"{hint}\n    " + "\n    ".join(missing))

    api_key = os.environ["LLM_API_KEY"].strip()
    if _PLACEHOLDER_KEY.match(api_key):
        raise ConfigError(
            f"{ENV_FILE} 里的 LLM_API_KEY 还是模板占位值（{_mask_secret(api_key)}）。\n"
            f"请打开该文件填入真实密钥。注意：该文件已被 .gitignore 排除，不会进仓库。"
        )

    # 相对路径统一锚定到项目根，避免"在哪个目录执行脚本"影响结果
    db_path = Path(os.environ.get("DB_PATH", "").strip() or _DEFAULT_DB_PATH)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    return Config(
        llm_api_key=api_key,
        llm_base_url=os.environ["LLM_BASE_URL"].strip().rstrip("/"),
        llm_model=os.environ["LLM_MODEL"].strip(),
        db_path=db_path,
    )


# 进程内缓存：.env 读一次即可，重复调用不产生额外 IO
_cached_config: Config | None = None


def get_config(*, reload: bool = False) -> Config:
    """获取全局配置（带缓存）。

    :param reload: 传 True 强制重新读取，便于测试或改了 .env 后热更新。
    :raises ConfigError: 配置缺失或非法时抛出，消息里含修复指引。
    """
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = _build_config()
    return _cached_config


if __name__ == "__main__":
    try:
        cfg = get_config()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        raise SystemExit(1)

    db_state = "已存在" if cfg.db_path.exists() else "尚未创建"
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"配置文件　：{ENV_FILE}（{'已加载' if ENV_FILE.exists() else '不存在'}）")
    print("-" * 52)
    for key, value in cfg.masked().items():
        print(f"  {key:<12} = {value}")
    print("-" * 52)
    print(f"数据库路径：{cfg.db_path}（{db_state}）")
