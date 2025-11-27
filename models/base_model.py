class BaseVLLM:
    """统一的视觉语言模型接口"""

    def analyze(self, screenshot_path: str, dom_summary: str, user_goal: str, current_url: str) -> dict:
        """
        输入网页截图和 DOM 信息，输出决定动作的 JSON。
        所有模型必须实现。
        """
        raise NotImplementedError("analyze() 未实现")
