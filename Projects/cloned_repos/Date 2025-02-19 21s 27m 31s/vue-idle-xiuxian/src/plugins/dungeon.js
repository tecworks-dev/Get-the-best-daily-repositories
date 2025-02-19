// 副本难度配置
const difficultyModifiers = {
    easy: { healthMod: 0.8, damageMod: 0.8, rewardMod: 0.8 },
    normal: { healthMod: 1.0, damageMod: 1.0, rewardMod: 1.0 },
    hard: { healthMod: 1.2, damageMod: 1.2, rewardMod: 1.5 },
    expert: { healthMod: 1.5, damageMod: 1.5, rewardMod: 2.0 }
}

// 装备品质
const equipmentQualities = {
    common: { name: '凡品', color: '#9e9e9e', statMod: 1.0, realmRequirement: 1 },  // 练气期
    uncommon: { name: '下品', color: '#4caf50', statMod: 1.2, realmRequirement: 5 },  // 练气中期
    rare: { name: '中品', color: '#2196f3', statMod: 1.5, realmRequirement: 10 },  // 筑基期
    epic: { name: '上品', color: '#9c27b0', statMod: 2.0, realmRequirement: 19 },  // 金丹期
    legendary: { name: '极品', color: '#ff9800', statMod: 2.5, realmRequirement: 28 },  // 元婴期
    mythic: { name: '仙品', color: '#e91e63', statMod: 3.0, realmRequirement: 37 }  // 化神期
}

// 装备类型
const equipmentTypes = {
    weapon: { name: '法器', slot: 'weapon', prefixes: ['九天', '太虚', '混沌', '玄天', '紫霄', '青冥', '赤炎', '幽冥'] },
    head: { name: '法冠', slot: 'head', prefixes: ['天灵', '玄冥', '紫金', '青玉', '赤霞', '幽月', '星辰', '云霄'] },
    body: { name: '法衣', slot: 'body', prefixes: ['九霄', '太素', '混元', '玄阳', '紫薇', '青龙', '赤凤', '幽冥'] },
    legs: { name: '法裤', slot: 'legs', prefixes: ['天罡', '玄武', '紫电', '青云', '赤阳', '幽灵', '星光', '云雾'] },
    feet: { name: '法靴', slot: 'feet', prefixes: ['天行', '玄风', '紫霞', '青莲', '赤焰', '幽影', '星步', '云踪'] },
    shoulder: { name: '护肩', slot: 'shoulder', prefixes: ['天护', '玄甲', '紫雷', '青锋', '赤羽', '幽岚', '星芒', '云甲'] },
    hands: { name: '法手', slot: 'hands', prefixes: ['天罗', '玄玉', '紫晶', '青钢', '赤金', '幽银', '星铁', '云纹'] },
    wrist: { name: '护腕', slot: 'wrist', prefixes: ['天绝', '玄铁', '紫玉', '青石', '赤铜', '幽钢', '星晶', '云纱'] },
    necklace: { name: '法链', slot: 'necklace', prefixes: ['天珠', '玄圣', '紫灵', '青魂', '赤心', '幽魄', '星魂', '云珠'] },
    ring1: { name: '法戒', slot: 'ring1', prefixes: ['天命', '玄命', '紫命', '青命', '赤命', '幽命', '星命', '云命'] },
    ring2: { name: '法戒', slot: 'ring2', prefixes: ['天道', '玄道', '紫道', '青道', '赤道', '幽道', '星道', '云道'] },
    belt: { name: '法带', slot: 'belt', prefixes: ['天系', '玄系', '紫系', '青系', '赤系', '幽系', '星系', '云系'] },
    artifact: { name: '法宝', slot: 'artifact', prefixes: ['天宝', '玄宝', '紫宝', '青宝', '赤宝', '幽宝', '星宝', '云宝'] }
}

// 装备基础属性
const equipmentBaseStats = {
    weapon: {
        attack: { name: '攻击', min: 10, max: 20 },
        critRate: { name: '暴击率', min: 0.05, max: 0.1 },
        critDamageBoost: { name: '暴击伤害', min: 0.1, max: 0.3 }
    },
    head: {
        defense: { name: '防御', min: 5, max: 10 },
        health: { name: '生命', min: 50, max: 100 },
        stunResist: { name: '抗眩晕', min: 0.05, max: 0.1 }
    },
    body: {
        defense: { name: '防御', min: 8, max: 15 },
        health: { name: '生命', min: 80, max: 150 },
        finalDamageReduce: { name: '最终减伤', min: 0.05, max: 0.1 }
    },
    legs: {
        defense: { name: '防御', min: 6, max: 12 },
        speed: { name: '速度', min: 5, max: 10 },
        dodgeRate: { name: '闪避率', min: 0.05, max: 0.1 }
    },
    feet: {
        defense: { name: '防御', min: 4, max: 8 },
        speed: { name: '速度', min: 8, max: 15 },
        dodgeRate: { name: '闪避率', min: 0.05, max: 0.1 }
    },
    shoulder: {
        defense: { name: '防御', min: 5, max: 10 },
        health: { name: '生命', min: 40, max: 80 },
        counterRate: { name: '反击率', min: 0.05, max: 0.1 }
    },
    hands: {
        attack: { name: '攻击', min: 5, max: 10 },
        critRate: { name: '暴击率', min: 0.03, max: 0.08 },
        comboRate: { name: '连击率', min: 0.05, max: 0.1 }
    },
    wrist: {
        defense: { name: '防御', min: 3, max: 8 },
        counterRate: { name: '反击率', min: 0.05, max: 0.1 },
        vampireRate: { name: '吸血率', min: 0.05, max: 0.1 }
    },
    necklace: {
        health: { name: '生命', min: 60, max: 120 },
        healBoost: { name: '强化治疗', min: 0.1, max: 0.2 },
        spiritRate: { name: '灵力获取', min: 0.1, max: 0.2 }
    },
    ring1: {
        attack: { name: '攻击', min: 5, max: 10 },
        critDamageBoost: { name: '暴击伤害', min: 0.1, max: 0.2 },
        finalDamageBoost: { name: '最终增伤', min: 0.05, max: 0.1 }
    },
    ring2: {
        defense: { name: '防御', min: 5, max: 10 },
        critDamageReduce: { name: '爆伤减免', min: 0.1, max: 0.2 },
        resistanceBoost: { name: '抗性提升', min: 0.05, max: 0.1 }
    },
    belt: {
        health: { name: '生命', min: 40, max: 80 },
        defense: { name: '防御', min: 4, max: 8 },
        combatBoost: { name: '战斗属性', min: 0.05, max: 0.1 }
    },
    artifact: {
        cultivationRate: { name: '修炼速率', min: 0.1, max: 0.3 },
        spiritRate: { name: '灵力获取', min: 0.1, max: 0.3 },
        luck: { name: '福缘', min: 0.1, max: 0.3 }
    }
}

// 特殊效果池
const specialEffects = [
    { id: 'spirit_enhance', name: '聚灵', description: '提升20%灵力获取', effect: (stats) => { stats.spiritRate *= 1.2 } },
    { id: 'cultivation_boost', name: '悟道', description: '提升15%修炼速度', effect: (stats) => { stats.cultivationRate *= 1.15 } },
    { id: 'combat_mastery', name: '战法', description: '提升15%战斗属性', effect: (stats) => { stats.combatBoost += 0.15 } },
    { id: 'divine_protection', name: '天护', description: '提升10%最终减伤', effect: (stats) => { stats.finalDamageReduce += 0.1 } },
    { id: 'spirit_resonance', name: '灵韵', description: '提升15%灵力上限', effect: (stats) => { stats.maxSpirit *= 1.15 } },
    { id: 'fortune_blessing', name: '福缘', description: '提升10%幸运值', effect: (stats) => { stats.luck *= 1.1 } },
    { id: 'immortal_body', name: '不朽', description: '提升20%生命上限', effect: (stats) => { stats.maxHealth *= 1.2 } },
    { id: 'divine_might', name: '神威', description: '提升10%最终伤害', effect: (stats) => { stats.finalDamageBoost += 0.1 } }
]

// 随机选项池
const roguelikeOptions = {
    common: [
        { id: 'heal', name: '灵力恢复', description: '恢复30%灵力', effect: (player) => { 
            if (player.spirit !== undefined && player.maxSpirit !== undefined) {
                player.spirit = Math.min(player.maxSpirit, player.spirit + player.maxSpirit * 0.3)
            }
        } },
        { id: 'small_buff', name: '小幅强化', description: '增加10%伤害', effect: (player) => { 
            if (player.combatAttributes) {
                player.combatAttributes.finalDamageBoost = (player.combatAttributes.finalDamageBoost || 0) + 0.1
            }
        } },
        { id: 'spirit_shield', name: '灵力护盾', description: '获得一层护盾，可抵挡一次伤害', effect: (player) => { 
            player.shield = (player.shield || 0) + 1
        } },
        { id: 'cultivation_boost', name: '悟道加持', description: '临时提升20%修炼速度', effect: (player) => { 
            player.cultivationRate = (player.cultivationRate || 1) * 1.2
        } },
        { id: 'spirit_gathering', name: '聚灵', description: '提升15%灵力获取', effect: (player) => { 
            player.spiritRate = (player.spiritRate || 1) * 1.15
        } },
        { id: 'defense_boost', name: '铁壁', description: '提升20%防御力', effect: (player) => { 
            if (player.baseAttributes) {
                player.baseAttributes.defense = (player.baseAttributes.defense || 5) * 1.2
            }
        } },
        { id: 'speed_boost', name: '疾风', description: '提升15%速度', effect: (player) => { 
            if (player.baseAttributes) {
                player.baseAttributes.speed = (player.baseAttributes.speed || 10) * 1.15
            }
        } },
        { id: 'health_regen', name: '生机', description: '每回合恢复5%生命值', effect: (player) => { 
            player.healthRegen = (player.healthRegen || 0) + 0.05
        } },
        { id: 'spirit_regen', name: '灵息', description: '每回合恢复3%灵力', effect: (player) => { 
            player.spiritRegen = (player.spiritRegen || 0) + 0.03
        } },
        { id: 'luck_boost', name: '福运', description: '提升10%幸运值', effect: (player) => { 
            player.luck = (player.luck || 1) * 1.1
        } }
    ],
    rare: [
        { id: 'double_damage', name: '伤害倍增', description: '本次副本伤害翻倍', effect: (player) => { 
            if (player.combatAttributes) {
                player.combatAttributes.finalDamageBoost = (player.combatAttributes.finalDamageBoost || 0) * 2
            }
        } },
        { id: 'special_item', name: '特殊装备', description: '获得一件随机品质装备', effect: (player) => { 
            if (player.items) {
                player.items.push(generateEquipment(player.level || 1))
            }
        } },
        { id: 'breakthrough_chance', name: '突破机缘', description: '获得一次突破机会', effect: (player) => { 
            player.breakthroughChance = (player.breakthroughChance || 0) + 1
        } },
        { id: 'spirit_blessing', name: '灵气祝福', description: '所有属性提升25%', effect: (player) => { 
            if (player.baseAttributes) {
                Object.keys(player.baseAttributes).forEach(key => {
                    player.baseAttributes[key] = (player.baseAttributes[key] || 0) * 1.25
                })
            }
        } },
        { id: 'combat_enlightenment', name: '战斗顿悟', description: '战斗经验获取翻倍', effect: (player) => { 
            player.combatExpRate = (player.combatExpRate || 1) * 2
        } },
        { id: 'crit_mastery', name: '会心', description: '暴击率提升30%，暴击伤害提升50%', effect: (player) => { 
            if (player.combatAttributes) {
                player.combatAttributes.critRate = (player.combatAttributes.critRate || 0) * 1.3
                player.specialAttributes.critDamageBoost = (player.specialAttributes.critDamageBoost || 0) * 1.5
            }
        } },
        { id: 'dodge_master', name: '无影', description: '闪避率提升40%', effect: (player) => { 
            if (player.combatAttributes) {
                player.combatAttributes.dodgeRate = (player.combatAttributes.dodgeRate || 0) * 1.4
            }
        } },
        { id: 'spirit_link', name: '灵脉', description: '攻击时有30%概率额外造成灵力伤害', effect: (player) => { 
            player.spiritDamageChance = 0.3
        } },
        { id: 'life_bond', name: '生命链接', description: '造成伤害时回复10%生命值', effect: (player) => { 
            player.lifeSteal = (player.lifeSteal || 0) + 0.1
        } },
        { id: 'realm_resonance', name: '境界共鸣', description: '基于当前境界提升20-50%属性', effect: (player) => { 
            player.realmBonus = (player.realmBonus || 0) + 0.2 + ((player.level || 1) * 0.01)
        } }
    ],
    epic: [
        { id: 'immortal', name: '不死之身', description: '一次致命伤害免疫', effect: (player) => { 
            player.immunityCount = (player.immunityCount || 0) + 1
        } },
        { id: 'ultimate_power', name: '极限突破', description: '属性临时提升50%', effect: (player) => { 
            player.tempStatBonus = (player.tempStatBonus || 0) + 0.5
        } },
        { id: 'realm_insight', name: '境界感悟', description: '立即获得大量修为', effect: (player) => { 
            if (player.cultivation !== undefined && player.maxCultivation !== undefined) {
                player.cultivation = Math.min(player.maxCultivation, player.cultivation + player.maxCultivation * 0.5)
            }
        } },
        { id: 'divine_protection', name: '天道庇护', description: '获得三层护盾', effect: (player) => { 
            player.shield = (player.shield || 0) + 3
        } },
        { id: 'legendary_item', name: '仙缘际遇', description: '获得一件极品或仙品装备', effect: (player) => { 
            if (player.items) {
                player.items.push(generateEquipment(player.level || 1, null, Math.random() < 0.7 ? 'legendary' : 'mythic'))
            }
        } },
        { id: 'heaven_blessing', name: '天地眷顾', description: '所有增益效果提升100%', effect: (player) => { 
            player.buffEffectiveness = (player.buffEffectiveness || 1) * 2
        } },
        { id: 'immortal_insight', name: '仙人顿悟', description: '获得一个随机特殊效果，持续整个副本', effect: (player) => { 
            if (typeof player.addRandomSpecialEffect === 'function') {
                player.addRandomSpecialEffect()
            }
        } }
    ]
}

// 生成随机装备
function generateEquipment(level, type = null, quality = null) {
    // 随机选择装备类型
    if (!type) {
        const types = Object.keys(equipmentTypes)
        type = types[Math.floor(Math.random() * types.length)]
    }
    // 随机选择品质，根据玩家等级调整概率
    if (!quality) {
        const qualities = Object.keys(equipmentQualities)
        const levelBonus = Math.min(0.2, level * 0.01) // 每级增加1%的高品质概率，最高20%
        const roll = Math.random() - levelBonus
        if (roll < 0.35) quality = 'common'
        else if (roll < 0.65) quality = 'uncommon'
        else if (roll < 0.82) quality = 'rare'
        else if (roll < 0.93) quality = 'epic'
        else if (roll < 0.98) quality = 'legendary'
        else quality = 'mythic'
    }
    // 检查境界要求
    const requiredRealm = equipmentQualities[quality].realmRequirement
    // 基础属性计算
    const baseStats = {}
    const qualityMod = equipmentQualities[quality].statMod
    const levelMod = 1 + (level * 0.1)
    Object.entries(equipmentBaseStats[type]).forEach(([stat, config]) => {
        const base = config.min + Math.random() * (config.max - config.min)
        const value = base * qualityMod * levelMod
        // 对百分比属性进行特殊处理
        if (['critRate', 'critDamageBoost', 'dodgeRate', 'vampireRate', 'finalDamageBoost', 'finalDamageReduce'].includes(stat)) {
            baseStats[stat] = Math.round(value * 100) / 100 // 保留两位小数
        } else {
            baseStats[stat] = Math.round(value)
        }
    })
    // 随机特殊效果
    const effectCount = quality === 'mythic' ? 2 : quality === 'legendary' ? 1 : 0
    const effects = []
    if (effectCount > 0) {
        const availableEffects = [...specialEffects]
        for (let i = 0; i < effectCount; i++) {
            if (availableEffects.length === 0) break
            const index = Math.floor(Math.random() * availableEffects.length)
            effects.push(availableEffects.splice(index, 1)[0])
        }
    }
    return {
        id: `${type}_${Date.now()}`,
        name: generateEquipmentName(type, quality),
        type: type,  // 确保设置正确的type属性
        slot: type,  // 添加slot属性，用于装备系统
        quality,
        level,
        requiredRealm,
        stats: baseStats,
        effects,
        qualityInfo: equipmentQualities[quality]
    }
}
// 生成装备名称
const generateEquipmentName = (type, quality) => {
    const typeInfo = equipmentTypes[type]
    const qualityInfo = equipmentQualities[quality]
    const prefix = typeInfo.prefixes[Math.floor(Math.random() * typeInfo.prefixes.length)]
    const suffixes = ['', '·真', '·极', '·道', '·天', '·仙', '·圣', '·神']
    const suffix = quality === 'mythic' ? suffixes[7] :
                  quality === 'legendary' ? suffixes[6] :
                  quality === 'epic' ? suffixes[5] :
                  quality === 'rare' ? suffixes[4] :
                  quality === 'uncommon' ? suffixes[3] :
                  suffixes[0]
    return `${prefix}${typeInfo.name}${suffix}`
}
// 获取随机选项
const getRandomOptions = (floor) => {
    // 基础概率设置
    let commonChance = 0.7
    let rareChance = 0.25
    let epicChance = 0.05
    // 根据层数调整概率
    if (floor % 10 === 0) { // 每10层提高史诗品质概率
        commonChance = 0.5
        rareChance = 0.3
        epicChance = 0.2 // 提高史诗品质概率到20%
    } else if (floor % 5 === 0) { // 每5层提高稀有品质概率
        commonChance = 0.5
        rareChance = 0.35 // 提高稀有品质概率到35%
        epicChance = 0.15
    }
    const count = 3
    const selected = []
    const usedIds = new Set()
    while (selected.length < count) {
        // 为每个选项独立随机决定品质
        const rand = Math.random()
        let pool = 'common'
        if (rand < epicChance) {
            pool = 'epic'
        } else if (rand < epicChance + rareChance) {
            pool = 'rare'
        }
        // 从选定的池中随机选择一个选项
        const options = roguelikeOptions[pool].filter(opt => !usedIds.has(opt.id))
        if (options.length > 0) {
            const index = Math.floor(Math.random() * options.length)
            const option = options[index]
            selected.push(option)
            usedIds.add(option.id)
        } else {
            // 如果当前品质池中没有可用选项，尝试其他品质池
            const allOptions = [
                ...roguelikeOptions.common,
                ...roguelikeOptions.rare,
                ...roguelikeOptions.epic
            ].filter(opt => !usedIds.has(opt.id))
            if (allOptions.length > 0) {
                const index = Math.floor(Math.random() * allOptions.length)
                const option = allOptions[index]
                selected.push(option)
                usedIds.add(option.id)
            }
        }
    }
    return selected
}

export {
    difficultyModifiers,
    equipmentQualities,
    equipmentTypes,
    equipmentBaseStats,
    specialEffects,
    roguelikeOptions,
    generateEquipment,
    getRandomOptions
}