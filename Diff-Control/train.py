# 导入基础库
import argparse             # 命令行参数解析 ⇦ 用于处理--config等参数
import logging              # 日志记录模块 ⇦ 用于训练过程监控
import os                   # 系统路径操作
import yaml                 # YAML配置文件解析
from sys import argv        # 获取命令行参数
from config import cfg      # 自定义配置对象 ⇦ 来自config.py的配置容器
import controlnet_engine, BCZ_engine, BCZ_LSTM_engine, prebuild_engine, tomato_engine  # 各模型引擎 ⇦ 不同算法实现

# 过滤警告信息 ⇦ 忽略弃用警告避免输出干扰
# mini_controlnet_engine, BCZ_LSTM_engine
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 配置日志系统 ⇦ 设置日志级别和格式
# $python train.py --config tensegrity_1.0.yaml
logging_kwargs = dict(
    level="INFO",                               # INFO级别日志
    format="%(asctime)s %(threadName)s %(levelname)s %(name)s - %(message)s",  # 含时间戳、线程名
    style="%",                                  # 传统格式符
)
logging.basicConfig(**logging_kwargs)           # 应用日志配置
logger = logging.getLogger("diffusion-pilicy")  # 创建专属logger ⇦ 标识日志来源


# 命令行参数 > YAML文件 > 代码默认值，例如--batch-size 128会覆盖.yaml中的train.batch_size
def parse_args():
    """命令行参数解析与配置合并"""
    parser = argparse.ArgumentParser()
    # 必须指定的配置文件路径 ⇦ 通过--config指定.yaml文件
    parser.add_argument("--config", help="configuration file path", required=True)
    # 可选的覆盖参数 ⇦ 命令行优先级高于配置文件
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()
    
    # 加载YAML配置 ⇦ 将.yaml内容合并到cfg对象
    config_file = args.config
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)        # ⇦ 核心配置合并方法
        
    # 命令行参数覆盖配置 ⇦ 仅当参数非空时生效
    if bool(args.batch_size):
        cfg.train.batch_size = args.batch_size
    if bool(args.num_epochs):
        cfg.train.num_epochs = args.num_epochs
    return cfg, config_file                     # ⇦ 返回最终配置和文件路径


def main():
    # 解析配置并锁定 ⇦ 防止训练过程中配置被修改
    cfg, config_file = parse_args()
    cfg.freeze()                                # ⇦ 冻结配置对象

    ####### check all the parameter settings #######
    # 日志输出完整配置 ⇦ 调试时确认参数正确性
    logger.info("{}".format(cfg))
    logger.info("check mode - {}".format(cfg.mode.mode))

    # 创建日志目录结构 ⇦ 保证实验结果存储
    # ./experiments/ 
    # └── duck_controlnet/
    #     └── summaries/  # TensorBoard日志
    # Create directory for logs and experiment name
    if not os.path.exists(cfg.train.log_directory):
        os.mkdir(cfg.train.log_directory)
    if not os.path.exists(os.path.join(cfg.train.log_directory, cfg.train.model_name)):
        os.mkdir(os.path.join(cfg.train.log_directory, cfg.train.model_name))
        os.mkdir(
            os.path.join(cfg.train.log_directory, cfg.train.model_name, "summaries")
        )# ⇦ 创建TensorBoard日志目录
    else:
        logger.warning(
            "This logging directory already exists: {}. Over-writing current files".format(
                os.path.join(cfg.train.log_directory, cfg.train.model_name)
            )
        )

    ####### start the training #######
    # 根据配置选择训练引擎 ⇦ 核心分支逻辑
    if cfg.mode.model_zoo == "controlnet":
        train_engine = controlnet_engine.Engine(args=cfg, logger=logger)  # ⇦ ControlNet专用引擎
    elif cfg.mode.model_zoo == "diffusion-model":
        train_engine = prebuild_engine.Engine(args=cfg, logger=logger)  # ⇦ 基础扩散模型
    elif cfg.mode.model_zoo == "BCZ":
        train_engine = BCZ_engine.Engine(args=cfg, logger=logger)
    elif cfg.mode.model_zoo == "BCZ_LSTM":
        train_engine = BCZ_LSTM_engine.Engine(args=cfg, logger=logger)
    elif cfg.mode.model_zoo == "tomato-model":
        train_engine = tomato_engine.Engine(args=cfg, logger=logger)

    # 执行训练/测试流程 ⇦ 模式分发控制
    if cfg.mode.mode == "train":
        train_engine.train()                    # ⇦ 正常训练流程
    elif cfg.mode.mode == "pretrain":
        train_engine.train()                    # ⇦ 预训练（通过parameter_path配置,可能加载部分参数）
    elif cfg.mode.mode == "test":
        train_engine.test()                     # ⇦ 加载检查点进行测试


if __name__ == "__main__":
    main()                                      # ⇦ 程序入口
