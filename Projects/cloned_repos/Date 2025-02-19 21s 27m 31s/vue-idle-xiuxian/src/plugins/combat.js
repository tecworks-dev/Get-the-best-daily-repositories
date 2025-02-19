// 战斗状态
const CombatState = {
    READY: 'ready',
    IN_PROGRESS: 'in_progress',
    VICTORY: 'victory',
    DEFEAT: 'defeat'
}

// 战斗类型
const CombatType = {
    NORMAL: 'normal', // 普通战斗
    BOSS: 'boss', // Boss战斗
    ELITE: 'elite' // 精英战斗
}

// 基础战斗属性
class CombatStats {
    constructor(base = {}) {
        // 基础属性
        this.health = base.health || 100
        this.maxHealth = base.maxHealth || 100
        this.damage = base.damage || 10
        this.defense = base.defense || 5
        this.speed = base.speed || 10
        // 战斗属性（百分比）
        this.critRate = base.critRate || 0.05 // 暴击率
        this.comboRate = base.comboRate || 0 // 连击率
        this.counterRate = base.counterRate || 0 // 反击率
        this.stunRate = base.stunRate || 0 // 眩晕率
        this.dodgeRate = base.dodgeRate || 0.05 // 闪避率
        this.vampireRate = base.vampireRate || 0 // 吸血率
        // 战斗抗性（百分比）
        this.critResist = base.critResist || 0 // 抗暴击
        this.comboResist = base.comboResist || 0 // 抗连击
        this.counterResist = base.counterResist || 0 // 抗反击
        this.stunResist = base.stunResist || 0 // 抗眩晕
        this.dodgeResist = base.dodgeResist || 0 // 抗闪避
        this.vampireResist = base.vampireResist || 0 // 抗吸血
        // 特殊属性（百分比）
        this.healBoost = base.healBoost || 0 // 强化治疗
        this.critDamageBoost = base.critDamageBoost || 0.5 // 强化爆伤
        this.critDamageReduce = base.critDamageReduce || 0 // 弱化爆伤
        this.finalDamageBoost = base.finalDamageBoost || 0 // 最终增伤
        this.finalDamageReduce = base.finalDamageReduce || 0 // 最终减伤
        this.combatBoost = base.combatBoost || 0 // 战斗属性提升
        this.resistanceBoost = base.resistanceBoost || 0 // 战斗抗性提升
    }

    // 计算最终伤害
    calculateDamage() {
        let damage = this.damage
        let isCrit = false
        let isCombo = false
        // 计算暴击
        const finalCritRate = Math.max(0, this.critRate * (1 + this.combatBoost))
        if (Math.random() < finalCritRate) {
            damage *= (1.5 + this.critDamageBoost)
            isCrit = true
        }
        // 计算连击
        const finalComboRate = Math.max(0, this.comboRate * (1 + this.combatBoost))
        if (Math.random() < finalComboRate) {
            damage *= 1.3
            isCombo = true
        }
        // 应用最终伤害加成
        damage *= (1 + this.finalDamageBoost)
        return { damage, isCrit, isCombo }
    }
    // 计算伤害减免
    calculateDamageReduction(incomingDamage, attackerStats) {
        let damage = incomingDamage
        // 应用防御减伤
        damage *= (100 / (100 + this.defense))
        // 如果是暴击伤害，应用暴击伤害减免
        if (attackerStats && attackerStats.isCrit) {
            damage *= (1 - this.critDamageReduce)
        }
        // 应用最终伤害减免
        damage *= (1 - this.finalDamageReduce)
        return damage
    }
}

// 根据境界等级计算属性加成
function calculateRealmBonus(realmLevel) {
    return 1 + (realmLevel * 0.2)  // 每个境界提升20%的属性
}
// 战斗实体基类
class CombatEntity {
    constructor(name, level, stats = {}, realm = '练气一层') {
        this.name = name
        this.level = level
        this.realm = realm
        // 应用境界加成
        const realmBonus = calculateRealmBonus(level)
        Object.keys(stats).forEach(key => {
            if (typeof stats[key] === 'number') {
                stats[key] *= realmBonus
            }
        })
        this.stats = new CombatStats(stats)
        this.currentHealth = this.stats.maxHealth
        this.effects = []
    }
    // 受到伤害
    takeDamage(amount, source) {
        // 闪避判定
        if (Math.random() < this.stats.dodgeRate) {
            return { dodged: true, damage: 0 }
        }
        // 计算实际伤害
        const reducedDamage = this.stats.calculateDamageReduction(amount)
        this.currentHealth = Math.max(0, this.currentHealth - reducedDamage)
        return {
            dodged: false,
            damage: reducedDamage,
            currentHealth: this.currentHealth,
            isDead: this.currentHealth <= 0
        }
    }
    // 恢复生命值
    heal(amount) {
        const oldHealth = this.currentHealth
        this.currentHealth = Math.min(this.stats.maxHealth, this.currentHealth + amount)
        return this.currentHealth - oldHealth
    }
    // 添加效果
    addEffect(effect) {
        this.effects.push(effect)
        effect.apply(this)
    }
    // 移除效果
    removeEffect(effectId) {
        const index = this.effects.findIndex(e => e.id === effectId)
        if (index >= 0) {
            const effect = this.effects[index]
            effect.remove(this)
            this.effects.splice(index, 1)
        }
    }
}
// 战斗管理器
class CombatManager {
    constructor(player, enemy, logCallback, type = CombatType.NORMAL) {
        this.player = player
        this.enemy = enemy
        this.type = type
        this.state = CombatState.READY
        this.round = 0
        this.log = []
        this.logCallback = logCallback
    }
    // 开始战斗
    start() {
        this.state = CombatState.IN_PROGRESS
        const startLog = `战斗开始：${this.player.name} VS ${this.enemy.name}`
        this.log.push(startLog)
        if (this.logCallback) this.logCallback(startLog)
        return this.state
    }
    // 执行回合
    executeTurn() {
        if (this.state !== CombatState.IN_PROGRESS) return null
        this.round++
        const results = []
        // 玩家攻击
        const playerAttack = this.player.stats.calculateDamage()
        const enemyResult = this.enemy.takeDamage(playerAttack.damage, this.player)
        // 记录玩家攻击日志
        let attackLog = `${this.player.name}对${this.enemy.name}发起攻击`
        if (enemyResult.dodged) {
            attackLog += `，被闪避了！`
        } else {
            attackLog += `，造成${enemyResult.damage.toFixed(1)}点伤害`
            if (playerAttack.isCrit) attackLog += `（暴击！）`
            if (playerAttack.isCombo) attackLog += `（连击！）`
        }
        this.log.push(attackLog)
        if (this.logCallback) this.logCallback(attackLog)
        results.push({
            attacker: this.player.name,
            defender: this.enemy.name,
            damage: enemyResult.damage,
            isCrit: playerAttack.isCrit,
            isCombo: playerAttack.isCombo,
            isDodged: enemyResult.dodged
        })
        // 检查敌人是否死亡
        if (enemyResult.isDead) {
            this.state = CombatState.VICTORY
            this.log.push(`${this.player.name}获得胜利！`)
            return { results, state: this.state }
        }
        // 敌人反击
        const enemyAttack = this.enemy.stats.calculateDamage()
        const playerResult = this.player.takeDamage(enemyAttack.damage, this.enemy)
        // 记录敌人攻击日志
        let enemyAttackLog = `${this.enemy.name}发起反击`
        if (playerResult.dodged) {
            enemyAttackLog += `，被闪避了！`
        } else {
            enemyAttackLog += `，造成${playerResult.damage.toFixed(1)}点伤害`
            if (enemyAttack.isCrit) enemyAttackLog += `（暴击！）`
            if (enemyAttack.isCombo) enemyAttackLog += `（连击！）`
        }
        this.log.push(enemyAttackLog)
        results.push({
            attacker: this.enemy.name,
            defender: this.player.name,
            damage: playerResult.damage,
            isCrit: enemyAttack.isCrit,
            isCombo: enemyAttack.isCombo,
            isDodged: playerResult.dodged
        })
        // 检查玩家是否死亡
        if (playerResult.isDead) {
            this.state = CombatState.DEFEAT
            this.log.push(`${this.enemy.name}获得胜利！`)
        }
        return { results, state: this.state }
    }

    // 获取战斗日志
    getCombatLog() {
        return this.log
    }
    // 处理战斗失败
    handleDefeat(playerStore, currentFloor) {
        // 记录失败层数
        playerStore.dungeonLastFailedFloor = currentFloor
        // 重置副本进度
        playerStore.dungeonProgress = 1
        // 掉落所有装备
        Object.keys(playerStore.equippedArtifacts).forEach(slot => {
            if (playerStore.equippedArtifacts[slot]) {
                playerStore.unequipArtifact(slot)
            }
        })
        // 清空装备背包
        playerStore.artifacts = []
        this.log.push('战斗失败！所有装备已掉落，副本进度已重置。')
        // this.log.push('战斗失败！副本进度已重置。')
        playerStore.saveData()
    }
}

// 生成敌人
function generateEnemy(level, type = CombatType.NORMAL, realm = '练气一层') {
    const baseStats = {
        // 基础属性
        health: 80 + level * 20,
        damage: 8 + level * 2,
        defense: 3 + level,
        speed: 5 + level,
        // 战斗属性（百分比）
        critRate: 0.05 + (level * 0.002),
        comboRate: 0.03 + (level * 0.002),
        counterRate: 0.03 + (level * 0.002),
        stunRate: 0.02 + (level * 0.001),
        dodgeRate: 0.05 + (level * 0.002),
        vampireRate: 0.02 + (level * 0.001),
        // 战斗抗性（百分比）
        critResist: 0.02 + (level * 0.001),
        comboResist: 0.02 + (level * 0.001),
        counterResist: 0.02 + (level * 0.001),
        stunResist: 0.02 + (level * 0.001),
        dodgeResist: 0.02 + (level * 0.001),
        vampireResist: 0.02 + (level * 0.001),
        // 特殊属性（百分比）
        healBoost: 0.05 + (level * 0.002),
        critDamageBoost: 0.2 + (level * 0.01),
        critDamageReduce: 0.1 + (level * 0.005),
        finalDamageBoost: 0.05 + (level * 0.002),
        finalDamageReduce: 0.05 + (level * 0.002),
        combatBoost: 0.03 + (level * 0.002),
        resistanceBoost: 0.03 + (level * 0.002)
    }
    // 根据类型调整属性
    switch (type) {
        case CombatType.ELITE:
            Object.keys(baseStats).forEach(key => {
                if (typeof baseStats[key] === 'number') {
                    if (key.includes('Rate') || key.includes('Resist') || key.includes('Boost') || key.includes('Reduce')) {
                        baseStats[key] = Math.min(0.8, baseStats[key] * 1.3) // 百分比属性最高限制在80%
                    } else {
                        baseStats[key] *= 1.5
                    }
                }
            })
            break
        case CombatType.BOSS:
            Object.keys(baseStats).forEach(key => {
                if (typeof baseStats[key] === 'number') {
                    if (key.includes('Rate') || key.includes('Resist') || key.includes('Boost') || key.includes('Reduce')) {
                        baseStats[key] = Math.min(0.9, baseStats[key] * 1.5) // 百分比属性最高限制在90%
                    } else {
                        baseStats[key] *= 2
                    }
                }
            })
            break
    }
    // 根据类型和等级生成敌人名称
    let enemyName = ''
    const normalNames = ['野狼', '山猪', '毒蛇', '黑熊', '猛虎', '恶狼', '巨蟒', '狂狮']
    const eliteNames = ['赤焰虎', '玄冰蟒', '紫电豹', '金刚猿', '幽冥狼', '碧水蛟', '雷霆鹰', '烈风豹']
    const bossNames = ['九尾天狐', '万年龙蟒', '太古神虎', '玄天冰凤', '幽冥魔龙', '混沌巨兽', '远古天蟒', '不死火凤']
    switch (type) {
        case CombatType.BOSS:
            enemyName = bossNames[Math.floor(level / 10) % bossNames.length]
            break
        case CombatType.ELITE:
            enemyName = eliteNames[Math.floor(level / 5) % eliteNames.length]
            break
        default:
            enemyName = normalNames[level % normalNames.length]
    }
    return new CombatEntity(
        enemyName,
        level,
        baseStats,
        realm
    )
}
export {
    CombatState,
    CombatType,
    CombatStats,
    CombatEntity,
    CombatManager,
    generateEnemy
}