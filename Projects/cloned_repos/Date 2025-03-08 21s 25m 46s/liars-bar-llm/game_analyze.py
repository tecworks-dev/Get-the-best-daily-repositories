import os
import json
from collections import defaultdict, Counter

def analyze_game_records(folder_path):
    # 初始化统计数据结构
    stats = {
        'wins': Counter(),
        'shots_fired': Counter(),
        'survival_points': Counter(),
        'matchups': defaultdict(lambda: defaultdict(int)),  # A和B之间的对决次数记录
        'win_counts': defaultdict(lambda: defaultdict(int))  # A对B的胜利次数
    }
    
    player_names = set()
    game_count = 0
    
    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                
            # 跳过没有赢家的游戏
            if game_data.get('winner') is None:
                continue
                
            game_count += 1
            
            # 记录玩家名称
            for player in game_data.get('player_names', []):
                player_names.add(player)
                
            # 统计获胜情况
            winner = game_data.get('winner')
            if winner:
                stats['wins'][winner] += 1
            
            # 分析每一轮的数据
            rounds = game_data.get('rounds', [])
            for round_data in rounds:
                # 统计开枪情况
                round_result = round_data.get('round_result', {})
                shooter = round_result.get('shooter_name')
                if shooter:
                    stats['shots_fired'][shooter] += 1
                
                # 分析挑战对决情况
                play_history = round_data.get('play_history', [])
                for play in play_history:
                    player = play.get('player_name')
                    next_player = play.get('next_player')
                    was_challenged = play.get('was_challenged')
                    
                    if was_challenged and next_player:
                        challenge_result = play.get('challenge_result')
                        
                        # 记录对决次数 - 只记录一个方向，避免重复计数
                        # 确保按照字母顺序记录，使得对决始终以相同方式计数
                        if player < next_player:
                            stats['matchups'][player][next_player] += 1
                        else:
                            stats['matchups'][next_player][player] += 1
                        
                        # 记录谁赢了这次对决
                        if challenge_result is True:  # 挑战成功，next_player赢
                            stats['win_counts'][next_player][player] += 1
                        elif challenge_result is False:  # 挑战失败，player赢
                            stats['win_counts'][player][next_player] += 1
            
            # 计算存活积分
            # 首先确定淘汰顺序
            elimination_order = []
            alive_players = set(game_data.get('player_names', []))
            
            for round_data in rounds:
                round_result = round_data.get('round_result', {})
                shooter = round_result.get('shooter_name')
                bullet_hit = round_result.get('bullet_hit')
                
                if shooter and bullet_hit and shooter in alive_players:
                    elimination_order.append(shooter)
                    alive_players.remove(shooter)
            
            # 将剩余存活的玩家按照游戏结束时的顺序添加到淘汰顺序中
            elimination_order.extend(alive_players)
            
            # 计算每个玩家的存活积分
            # 如果有n个玩家，第一个淘汰的玩家得0分，第二个得1分，以此类推
            for i, player in enumerate(elimination_order):
                if i > 0:  # 第一个淘汰的不得分
                    stats['survival_points'][player] += i
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # 计算对决胜率
    win_rates = {}
    for player in player_names:
        win_rates[player] = {}
        for opponent in player_names:
            if player != opponent:
                # 确定配对的正确顺序以获取总对决次数
                if player < opponent:
                    total_matchups = stats['matchups'][player][opponent]
                else:
                    total_matchups = stats['matchups'][opponent][player]
                
                if total_matchups > 0:
                    wins = stats['win_counts'][player][opponent]
                    win_rates[player][opponent] = wins / total_matchups
                else:
                    win_rates[player][opponent] = 0
    
    return stats, win_rates, game_count, player_names

def print_statistics(stats, win_rates, game_count, player_names):
    players = sorted(list(player_names))
    
    print(f"总计分析了 {game_count} 场游戏")
    print("\n胜利局数统计:")
    for player in players:
        wins = stats['wins'][player]
        win_percentage = (wins / game_count) * 100 if game_count > 0 else 0
        print(f"{player}: {wins} 场 ({win_percentage:.1f}%)")
    
    print("\n开枪次数统计:")
    for player in players:
        print(f"{player}: {stats['shots_fired'][player]} 次")
    
    print("\n存活积分统计:")
    for player in players:
        points = stats['survival_points'][player]
        avg_points = points / game_count if game_count > 0 else 0
        print(f"{player}: {points} 分 (平均每局 {avg_points:.2f} 分)")
    
    print("\n对位对决胜率:")
    print(f"{'玩家 vs 对手':<25} {'对决次数':<10} {'胜利次数':<10} {'胜率':<10}")
    print("-" * 55)
    
    for player in players:
        for opponent in players:
            if player != opponent:
                # 获取正确顺序的对决总次数
                if player < opponent:
                    matchups = stats['matchups'][player][opponent]
                else:
                    matchups = stats['matchups'][opponent][player]
                
                wins = stats['win_counts'][player][opponent]
                win_rate = win_rates[player][opponent] * 100
                
                print(f"{player} vs {opponent:<10} {matchups:<10} {wins:<10} {win_rate:.1f}%")

if __name__ == "__main__":
    folder_path = "game_records"  # 替换为实际的文件夹路径
    stats, win_rates, game_count, player_names = analyze_game_records(folder_path)
    print_statistics(stats, win_rates, game_count, player_names)