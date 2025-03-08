import json
import os
from itertools import combinations
from collections import defaultdict

def format_challenge_event(history_item, round_data, player_states, game_id):
    """
    将单次对决事件格式化为可读文本，包含更多细节
    参数:
        history_item: 包含对决信息的字典
        round_data: 当前轮次的完整数据
        player_states: 所有玩家的初始状态
        game_id: 游戏标识符
    返回:
        格式化后的对决文本描述
    """
    # 提取对决双方信息
    player = history_item['player_name']
    next_player = history_item['next_player']
    
    # 查找玩家初始状态
    player_initial_state = None
    next_player_initial_state = None
    for state in round_data['player_initial_states']:
        if state['player_name'] == player:
            player_initial_state = state
        elif state['player_name'] == next_player:
            next_player_initial_state = state
    
    # 构建详细的对决记录
    output = []
    
    # 添加游戏标识
    output.append(f"游戏ID: {game_id}")
    
    # 添加出牌玩家信息
    output.append(f"出牌方 ({player}):")
    output.append(f"初始手牌: {', '.join(player_initial_state['initial_hand'])}")
    output.append(f"打出牌: {', '.join(history_item['played_cards'])}")
    output.append(f"剩余手牌: {', '.join(history_item['remaining_cards'])}")
    if 'play_reason' in history_item and history_item['play_reason']:
        output.append(f"出牌理由: {history_item['play_reason']}")
    if 'behavior' in history_item and history_item['behavior']:
        output.append(f"出牌表现: {history_item['behavior']}")
    
    # 添加质疑相关信息
    output.append(f"\n质疑方 ({next_player}):")
    if next_player_initial_state:
        output.append(f"初始手牌: {', '.join(next_player_initial_state['initial_hand'])}")
    
    if history_item['was_challenged']:
        output.append(f"发起质疑")
        if 'challenge_reason' in history_item and history_item['challenge_reason']:
            output.append(f"质疑理由: {history_item['challenge_reason']}")
        result_text = "成功" if history_item['challenge_result'] else "失败"
        output.append(f"质疑结果: {result_text}")
    else:
        output.append("选择不质疑")
        if 'challenge_reason' in history_item and history_item['challenge_reason']:
            output.append(f"不质疑理由: {history_item['challenge_reason']}")
    
    # 添加额外空行以提高可读性
    output.append("")
    
    return "\n".join(output)

def extract_matchups(game_data, game_id):
    """
    从游戏数据中提取所有玩家间的详细对决记录
    参数:
        game_data: 完整的游戏数据字典
        game_id: 游戏标识符
    返回:
        包含所有配对对决记录的字典
    """
    # 获取所有玩家名称并创建对决配对
    players = game_data['player_names']
    matchups = defaultdict(list)
    
    # 遍历处理每一轮的数据
    for round_data in game_data['rounds']:
        round_id = round_data['round_id']
        target_card = round_data['target_card']
        
        # 处理每一次出牌
        for play in round_data['play_history']:
            player = play['player_name']
            next_player = play['next_player']
            
            # 只记录发生质疑的对决
            if play['was_challenged']:
                matchup_key = '_vs_'.join(sorted([player, next_player]))
                
                # 添加轮次信息
                round_info = [
                    f"第 {round_id} 轮对决",
                    f"目标牌: {target_card}",
                    "=" * 40,
                    ""
                ]
                
                # 添加详细的对决记录
                challenge_text = format_challenge_event(play, round_data, round_data['player_initial_states'], game_id)
                
                # 合并所有信息
                full_text = "\n".join(round_info) + challenge_text
                
                matchups[matchup_key].append(full_text)
                
    return matchups

def save_matchups_to_files(all_matchups, output_dir):
    """
    将所有游戏的对决记录合并保存到单独的文件中
    参数:
        all_matchups: 包含所有游戏所有配对对决记录的字典
        output_dir: 输出文件夹路径
    """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存每对玩家的对决记录
    for matchup_key, interactions in all_matchups.items():
        if interactions:
            # 在输出文件夹中创建文件
            filename = os.path.join(output_dir, f"{matchup_key}_detailed_matchups.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"{matchup_key.replace('_vs_', ' 对阵 ')} 的详细对决记录\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n\n".join(interactions))
                # 在文件末尾添加统计信息
                f.write(f"\n\n总计对决次数: {len(interactions)}\n")

def process_all_json_files(input_dir, output_dir):
    """
    处理指定文件夹中的所有JSON文件，并合并相同玩家对的对决记录
    参数:
        input_dir: 输入文件夹路径（包含JSON文件）
        output_dir: 输出文件夹路径
    """
    # 确保输入文件夹存在
    if not os.path.exists(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 不存在")
        return
    
    # 遍历所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print(f"警告：在 '{input_dir}' 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 用于存储所有游戏的对决记录
    all_matchups = defaultdict(list)
    
    # 处理每个JSON文件
    for json_file in json_files:
        print(f"正在处理: {json_file}")
        file_path = os.path.join(input_dir, json_file)
        
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # 使用文件名作为游戏ID
            game_id = os.path.splitext(json_file)[0]
            
            # 提取对决记录
            game_matchups = extract_matchups(game_data, game_id)
            
            # 合并到总记录中
            for key, value in game_matchups.items():
                all_matchups[key].extend(value)
            
            print(f"已成功处理 {json_file}")
            
        except Exception as e:
            print(f"处理 {json_file} 时出错: {str(e)}")
    
    # 保存合并后的对决记录
    save_matchups_to_files(all_matchups, output_dir)
    print("所有对决记录已合并保存")

# 主程序开始

# 定义输入和输出文件夹
input_dir = "game_records"  # 包含JSON文件的文件夹
output_dir = "matchup_records"  # 输出文件夹

# 处理所有JSON文件
process_all_json_files(input_dir, output_dir)