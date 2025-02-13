from eclipse.result import GoalResult


class InvalidType(Exception):
    pass


class ToolError(Exception):
    pass


class StopEclipse(Exception):

    def __init__(self, message: str, goal_result: GoalResult):
        self.message = message
        self.goal_result = goal_result

    def __str__(self):
        return f"StopEclipse: {self.message}"
