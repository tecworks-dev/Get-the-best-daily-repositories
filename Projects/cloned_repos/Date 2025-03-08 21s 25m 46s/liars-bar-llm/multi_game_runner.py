from game import Game
from typing import Dict, List
import argparse

class MultiGameRunner:
    def __init__(self, player_configs: List[Dict[str, str]], num_games: int = 10):
        """初始化多局游戏运行器
        
        Args:
            player_configs: 玩家配置列表
            num_games: 要运行的游戏局数
        """
        self.player_configs = player_configs
        self.num_games = num_games

    def run_games(self) -> None:
        """运行指定数量的游戏"""
        for game_num in range(1, self.num_games + 1):
            print(f"\n=== 开始第 {game_num}/{self.num_games} 局游戏 ===")
            
            # 创建并运行新游戏
            game = Game(self.player_configs)
            game.start_game()
            
            print(f"第 {game_num} 局游戏结束")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行多局AI对战游戏',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-n', '--num-games',
        type=int,
        default=10,
        help='要运行的游戏局数 (默认: 10)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 配置玩家信息, 其中model为你通过API调用的模型名称
    player_configs = [
        {"name": "DeepSeek", "model": "deepseek-r1"},
        {"name": "ChatGPT", "model": "o3-mini"},
        {"name": "Claude", "model": "claude-3.7-sonnet"},
        {"name": "Gemini", "model": "gemini-2.0-flash-thinking"}
    ]
    
    # 创建并运行多局游戏
    runner = MultiGameRunner(player_configs, num_games=args.num_games)
    runner.run_games()