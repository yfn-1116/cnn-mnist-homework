def build_scheduler(optimizer, config: dict):
    sched_cfg = config.get("scheduler", {})
    if not sched_cfg.get("use", False):
        return None

    # 这里预留后续扩展
    return None
