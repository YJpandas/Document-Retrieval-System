#!/usr/bin/env python
"""
文档检索系统Web服务启动文件
"""
import os
import argparse
from typing import Dict, Any

from core.utils.logger import get_logger, setup_logging
from core.utils.config_loader import config_loader
from core.utils.gpu_accelerator import gpu_accelerator
from core.retrieval.retrieval_core import retrieval_core
from gui.app import create_app

# 设置日志
setup_logging()
logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="文档检索系统Web服务")
    parser.add_argument("--host", type=str, help="服务器主机地址")
    parser.add_argument("--port", type=int, help="服务器端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--rebuild-index", action="store_true", help="启动前重建索引")
    parser.add_argument("--gpu-disabled", action="store_true", help="禁用GPU加速")
    return parser.parse_args()


def print_gpu_info():
    """打印GPU信息"""
    gpu_info = gpu_accelerator.get_device_info()
    
    logger.info("===== GPU加速信息 =====")
    if gpu_info.get("gpu_available", False):
        logger.info(f"GPU状态: {'启用' if gpu_info.get('use_gpu', False) else '禁用'}")
        logger.info(f"设备: {gpu_info.get('device', 'unknown')}")
        if "gpu_name" in gpu_info:
            logger.info(f"GPU型号: {gpu_info['gpu_name']}")
        if "gpu_memory_total" in gpu_info:
            logger.info(f"GPU总内存: {gpu_info['gpu_memory_total']:.2f} GB")
        if "mixed_precision" in gpu_info:
            logger.info(f"混合精度: {'启用' if gpu_info['mixed_precision'] else '禁用'}")
    else:
        logger.info("GPU状态: 不可用")
    logger.info("=======================")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = config_loader.get_config("app_config")
    server_config = config.get("server", {})
    
    # 设置服务器参数
    host = args.host or server_config.get("host", "127.0.0.1")
    port = args.port or server_config.get("port", 5000)
    debug = args.debug or server_config.get("debug", False)
    
    # 初始化GPU（可以通过参数禁用）
    if args.gpu_disabled:
        logger.info("通过命令行参数禁用GPU加速")
        gpu_accelerator.use_gpu = False
    
    # 打印GPU信息
    print_gpu_info()
    
    # 初始化检索核心
    logger.info("初始化检索核心...")
    retrieval_core.initialize()
    
    # 如果需要，重建索引
    if args.rebuild_index:
        logger.info("正在重建索引...")
        result = retrieval_core.refresh_index(rebuild=True)
        if result:
            logger.info("索引重建成功")
        else:
            logger.error("索引重建失败")
    
    # 创建并运行Flask应用
    logger.info(f"启动Web服务: http://{host}:{port}")
    app = create_app(config)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()