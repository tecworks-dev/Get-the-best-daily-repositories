// 成就系统配置
export const achievements = {
    // 秘境探索成就
    dungeon_explore: [
        { id: 'dungeon_1', name: '初探秘境', description: '首次进入秘境', condition: (player) => player.dungeonTotalRuns >= 1, reward: { spirit: 200 } },
        { id: 'dungeon_2', name: '秘境探索者', description: '通过第10层秘境', condition: (player) => player.dungeonHighestFloor >= 10, reward: { spirit: 1000, damage: 1.1 } },
        { id: 'dungeon_3', name: '秘境征服者', description: '通过第50层秘境', condition: (player) => player.dungeonHighestFloor >= 50, reward: { spirit: 5000, damage: 1.2 } },
        { id: 'dungeon_4', name: '秘境王者', description: '通过第100层秘境', condition: (player) => player.dungeonHighestFloor >= 100, reward: { spirit: 20000, damage: 1.5 } },
        { id: 'dungeon_5', name: '秘境之主', description: '通过第200层秘境', condition: (player) => player.dungeonHighestFloor >= 200, reward: { spirit: 50000, damage: 2 } }
    ],
    
    // 秘境战斗成就
    dungeon_combat: [
        { id: 'dungeon_combat_1', name: '初战告捷', description: '击杀10个普通敌人', condition: (player) => player.dungeonTotalKills >= 10, reward: { spirit: 500 } },
        { id: 'dungeon_combat_2', name: '百战不殆', description: '击杀100个普通敌人', condition: (player) => player.dungeonTotalKills >= 100, reward: { spirit: 2000, defense: 1.1 } },
        { id: 'dungeon_combat_3', name: '精英猎手', description: '击杀50个精英敌人', condition: (player) => player.dungeonEliteKills >= 50, reward: { spirit: 5000, defense: 1.2 } },
        { id: 'dungeon_combat_4', name: 'BOSS终结者', description: '击杀10个BOSS', condition: (player) => player.dungeonBossKills >= 10, reward: { spirit: 10000, defense: 1.5 } },
        { id: 'dungeon_combat_5', name: '无敌战神', description: '总击杀数达到1000', condition: (player) => player.dungeonTotalKills >= 1000, reward: { spirit: 30000, defense: 2 } }
    ],

    // 修为成就
    cultivation: [
        { id: 'cultivation_1', name: '初入修仙', description: '开始踏上修仙之路', condition: (player) => player.totalCultivationTime > 0, reward: { spirit: 100 } },
        { id: 'cultivation_2', name: '勤修苦练', description: '累计修炼时间达到1小时', condition: (player) => player.totalCultivationTime >= 3600, reward: { spirit: 500 } },
        { id: 'cultivation_3', name: '修炼成痴', description: '累计修炼时间达到24小时', condition: (player) => player.totalCultivationTime >= 86400, reward: { spirit: 2000 } },
        { id: 'cultivation_4', name: '道心稳固', description: '累计修炼时间达到7天', condition: (player) => player.totalCultivationTime >= 604800, reward: { spirit: 5000, spiritRate: 1.1 } },
        { id: 'cultivation_5', name: '修仙百年', description: '累计修炼时间达到30天', condition: (player) => player.totalCultivationTime >= 2592000, reward: { spirit: 10000, spiritRate: 1.2 } }
    ],
    
    // 突破成就
    breakthrough: [
        { id: 'breakthrough_1', name: '初窥门径', description: '首次突破', condition: (player) => player.breakthroughCount >= 1, reward: { spirit: 200 } },
        { id: 'breakthrough_2', name: '修炼有成', description: '突破次数达到10次', condition: (player) => player.breakthroughCount >= 10, reward: { spirit: 1000 } },
        { id: 'breakthrough_3', name: '道法自然', description: '突破次数达到50次', condition: (player) => player.breakthroughCount >= 50, reward: { spirit: 5000 } },
        { id: 'breakthrough_4', name: '登峰造极', description: '突破次数达到100次', condition: (player) => player.breakthroughCount >= 100, reward: { spirit: 10000, spiritRate: 1.2 } },
        { id: 'breakthrough_5', name: '问鼎巅峰', description: '达到化神境界', condition: (player) => player.level >= 37, reward: { spirit: 50000, spiritRate: 1.5 } }
    ],
    
    // 探索成就
    exploration: [
        { id: 'exploration_1', name: '初探世界', description: '首次进行探索', condition: (player) => player.explorationCount >= 1, reward: { spirit: 100 } },
        { id: 'exploration_2', name: '四处游历', description: '探索次数达到100次', condition: (player) => player.explorationCount >= 100, reward: { spirit: 1000 } },
        { id: 'exploration_3', name: '游历天下', description: '探索次数达到1000次', condition: (player) => player.explorationCount >= 1000, reward: { spirit: 5000 } },
        { id: 'exploration_4', name: '寻宝达人', description: '获得100件物品', condition: (player) => player.itemsFound >= 100, reward: { spirit: 2000 } },
        { id: 'exploration_5', name: '机缘深厚', description: '触发100次随机事件', condition: (player) => player.eventTriggered >= 100, reward: { spirit: 3000, luck: 1.1 } }
    ],
    
    // 收集成就
    collection: [
        { id: 'collection_1', name: '初识灵草', description: '收集首株灵草', condition: (player) => player.herbs.length >= 1, reward: { spirit: 100 } },
        { id: 'collection_2', name: '灵草收藏家', description: '收集10种不同灵草', condition: (player) => new Set(player.herbs.map(h => h.id)).size >= 10, reward: { spirit: 1000 } },
        { id: 'collection_3', name: '灵草大师', description: '收集所有种类灵草', condition: (player) => new Set(player.herbs.map(h => h.id)).size >= 20, reward: { spirit: 3000, herbRate: 1.2 } },
        { id: 'collection_4', name: '仙品收藏', description: '获得首个仙品灵草', condition: (player) => player.herbs.some(h => h.quality === 'legendary'), reward: { spirit: 2000 } },
        { id: 'collection_5', name: '灵草园主', description: '拥有100株灵草', condition: (player) => player.herbs.length >= 100, reward: { spirit: 5000, herbRate: 1.5 } }
    ],
    
    // 资源成就
    resources: [
        { id: 'resources_1', name: '初获灵石', description: '获得首枚灵石', condition: (player) => player.spiritStones >= 1, reward: { spirit: 100 } },
        { id: 'resources_2', name: '小有积蓄', description: '灵石数量达到1000', condition: (player) => player.spiritStones >= 1000, reward: { spirit: 1000 } },
        { id: 'resources_3', name: '富甲一方', description: '灵石数量达到10000', condition: (player) => player.spiritStones >= 10000, reward: { spirit: 5000 } },
        { id: 'resources_4', name: '富可敌国', description: '灵石数量达到100000', condition: (player) => player.spiritStones >= 100000, reward: { spirit: 20000, spiritRate: 1.3 } },
        { id: 'resources_5', name: '坐拥灵山', description: '灵石数量达到1000000', condition: (player) => player.spiritStones >= 1000000, reward: { spirit: 100000, spiritRate: 2 } }
    ],
    
    // 炼丹成就
    alchemy: [
        { id: 'alchemy_1', name: '初识丹道', description: '首次成功炼制丹药', condition: (player) => player.pillsCrafted >= 1, reward: { spirit: 200 } },
        { id: 'alchemy_2', name: '丹道小成', description: '成功炼制10颗丹药', condition: (player) => player.pillsCrafted >= 10, reward: { spirit: 1000 } },
        { id: 'alchemy_3', name: '丹道精通', description: '成功炼制100颗丹药', condition: (player) => player.pillsCrafted >= 100, reward: { spirit: 5000, alchemyRate: 1.2 } },
        { id: 'alchemy_4', name: '丹道大师', description: '获得所有丹方', condition: (player) => player.unlockedPillRecipes >= 8, reward: { spirit: 10000, alchemyRate: 1.5 } },
        { id: 'alchemy_5', name: '丹道宗师', description: '炼制1000颗丹药', condition: (player) => player.pillsCrafted >= 1000, reward: { spirit: 50000, alchemyRate: 2 } }
    ]
}

// 检查成就完成情况并发放奖励
export const checkAchievements = (player) => {
    const completedAchievements = []
    
    // 遍历所有成就类别
    Object.values(achievements).forEach(category => {
        category.forEach(achievement => {
            // 检查成就是否已完成且未记录
            if (achievement.condition(player) && 
                !player.completedAchievements?.includes(achievement.id)) {
                completedAchievements.push(achievement)
                
                // 添加到已完成成就列表
                if (!player.completedAchievements) {
                    player.completedAchievements = []
                }
                player.completedAchievements.push(achievement.id)
                
                // 发放成就奖励
                if (achievement.reward) {
                    if (achievement.reward.spirit) {
                        player.spirit += achievement.reward.spirit
                    }
                    if (achievement.reward.spiritRate) {
                        player.spiritRate *= achievement.reward.spiritRate
                    }
                    if (achievement.reward.herbRate) {
                        player.herbRate = (player.herbRate || 1) * achievement.reward.herbRate
                    }
                    if (achievement.reward.alchemyRate) {
                        player.alchemyRate = (player.alchemyRate || 1) * achievement.reward.alchemyRate
                    }
                    if (achievement.reward.luck) {
                        player.luck = (player.luck || 1) * achievement.reward.luck
                    }
                }
                
                // 保存玩家数据
                player.saveData()
            }
        })
    })
    
    return completedAchievements
}