<script setup>
import { ref, computed } from 'vue'
import { usePlayerStore } from '../stores/player'
import { CombatManager, CombatEntity, generateEnemy, CombatType } from '../plugins/combat'
import { generateEquipment, getRandomOptions } from '../plugins/dungeon'
import { useMessage } from 'naive-ui'
import LogPanel from '../components/LogPanel.vue'

const playerStore = usePlayerStore()
const message = useMessage()
const logRef = ref(null)

// 副本状态
const dungeonState = ref({
    floor: 0,
    inCombat: false,
    showingOptions: false,
    currentOptions: [],
    combatManager: null
})

// 当前战斗日志
const combatLog = ref([])

// 装备类型配置
const equipmentTypes = {
    weapon: '武器',
    head: '头部',
    body: '衣服',
    legs: '裤子',
    feet: '鞋子',
    shoulder: '肩甲',
    hands: '手套',
    wrist: '护腕',
    necklace: '项链',
    ring1: '戒指1',
    ring2: '戒指2',
    belt: '腰带',
    artifact: '法宝'
}

// 获取装备名称
const getEquipmentName = (type) => {
    return equipmentTypes[type] || type
}

// 根据选项类型获取颜色
const getOptionColor = (floor) => {
    if (floor % 10 === 0) return '#e91e63' // 史诗
    if (floor % 5 === 0) return '#2196f3'  // 稀有
    return '#4caf50' // 普通
}

// 创建玩家战斗实体
const createPlayerEntity = () => {
    // 基础属性
    const baseStats = {
        health: playerStore.baseAttributes.health,
        damage: playerStore.baseAttributes.attack,
        defense: playerStore.baseAttributes.defense,
        speed: playerStore.baseAttributes.speed,
        // 战斗属性
        critRate: playerStore.combatAttributes.critRate,
        comboRate: playerStore.combatAttributes.comboRate,
        counterRate: playerStore.combatAttributes.counterRate,
        stunRate: playerStore.combatAttributes.stunRate,
        dodgeRate: playerStore.combatAttributes.dodgeRate,
        vampireRate: playerStore.combatAttributes.vampireRate,
        // 战斗抗性
        critResist: playerStore.combatResistance.critResist,
        comboResist: playerStore.combatResistance.comboResist,
        counterResist: playerStore.combatResistance.counterResist,
        stunResist: playerStore.combatResistance.stunResist,
        dodgeResist: playerStore.combatResistance.dodgeResist,
        vampireResist: playerStore.combatResistance.vampireResist,
        // 特殊属性
        healBoost: playerStore.specialAttributes.healBoost,
        critDamageBoost: playerStore.specialAttributes.critDamageBoost,
        critDamageReduce: playerStore.specialAttributes.critDamageReduce,
        finalDamageBoost: playerStore.specialAttributes.finalDamageBoost,
        finalDamageReduce: playerStore.specialAttributes.finalDamageReduce,
        combatBoost: playerStore.specialAttributes.combatBoost,
        resistanceBoost: playerStore.specialAttributes.resistanceBoost,
        // 其他属性
        spiritDamage: playerStore.spirit * 0.1,
        maxHealth: playerStore.baseAttributes.health
    }

    return new CombatEntity(playerStore.name, playerStore.level, baseStats, playerStore.realm)
}

// 开始新的副本
const startDungeon = () => {
    dungeonState.value = {
        floor: 0,
        inCombat: false,
        showingOptions: false,
        currentOptions: [],
        combatManager: null
    }
    playerStore.dungeonTotalRuns++  // 增加总探索次数
    nextFloor()
}

// 进入下一层
const nextFloor = () => {
    dungeonState.value.floor++
    const floor = dungeonState.value.floor
    
    // 检查是否需要显示选项
    if (floor === 1 || floor % 5 === 0) {
        showOptions()
        return
    }
    
    startCombat()
}

// 显示随机选项
const showOptions = () => {
    dungeonState.value.showingOptions = true
    dungeonState.value.currentOptions = getRandomOptions(dungeonState.value.floor)
}

// 选择选项
const selectOption = (option) => {
    option.effect(playerStore)
    message.success(`选择了：${option.name}`)
    dungeonState.value.showingOptions = false
    dungeonState.value.currentOptions = []
    startCombat()
}

// 开始战斗
const startCombat = () => {
    const floor = dungeonState.value.floor
    const isBossFloor = floor % 10 === 0
    const isEliteFloor = floor % 5 === 0
    
    const enemyType = isBossFloor ? CombatType.BOSS : isEliteFloor ? CombatType.ELITE : CombatType.NORMAL
    
    // 创建玩家战斗实体，并应用所有增益效果
    const playerEntity = createPlayerEntity()
    
    // 创建敌人
    const enemy = generateEnemy(floor, enemyType)
    
    // 创建战斗管理器
    dungeonState.value.combatManager = new CombatManager(
        playerEntity,
        enemy,
        (log) => {
            if (logRef.value) {
                logRef.value.addLog(log)
            }
        }
    )
    
    dungeonState.value.inCombat = true
    dungeonState.value.combatManager.start() // 初始化战斗状态
    autoCombat() // 开始自动战斗
}

// 自动战斗
const autoCombat = async () => {
    while (dungeonState.value.inCombat) {
        const result = dungeonState.value.combatManager.executeTurn()
        if (!result) break
        
        // 更新战斗日志
        result.results.forEach(turn => {
            let message = `${turn.attacker} 对 ${turn.defender} 造成了 ${turn.damage.toFixed(1)} 点伤害`
            if (turn.isCrit) message += '（暴击！）'
            if (turn.isDodged) message = `${turn.defender} 闪避了 ${turn.attacker} 的攻击！`
            logRef.value?.addLog('info', message)
        })
        
        // 检查战斗是否结束
        if (result.state === 'victory') {
            handleVictory()
            break
        } else if (result.state === 'defeat') {
            handleDefeat()
            break
        }
        
        // 添加延迟使战斗动画更流畅
        await new Promise(resolve => setTimeout(resolve, 1000))
    }
}

// 处理胜利
const handleVictory = () => {
    dungeonState.value.inCombat = false
    message.success(`击败了第 ${dungeonState.value.floor} 层的敌人！`)
    
    // 更新统计数据
    playerStore.dungeonTotalKills++
    if (dungeonState.value.floor % 10 === 0) {
        playerStore.dungeonBossKills++
    } else if (dungeonState.value.floor % 5 === 0) {
        playerStore.dungeonEliteKills++
    }
    
    // 更新最高层数记录
    if (dungeonState.value.floor > playerStore.dungeonHighestFloor) {
        playerStore.dungeonHighestFloor = dungeonState.value.floor
    }
    
    // 获得奖励
    const rewards = generateRewards()
    rewards.forEach(reward => {
        if (reward.type === 'equipment') {
            playerStore.addEquipment(reward.item)
            message.success(`获得了 ${reward.item.name}！`)
        } else if (reward.type === 'spirit_stones') {
            playerStore.spiritStones += reward.amount
            message.success(`获得了 ${reward.amount} 灵石！`)
        }
        playerStore.dungeonTotalRewards++
    })
    
    // 进入下一层
    nextFloor()
}

// 处理失败
const handleDefeat = () => {
    dungeonState.value.inCombat = false
    message.error(`在第 ${dungeonState.value.floor} 层被击败了...`)
    playerStore.dungeonDeathCount++
    // 调用战斗管理器的handleDefeat方法，传递当前层数
    dungeonState.value.combatManager.handleDefeat(playerStore, dungeonState.value.floor)
}

// 生成奖励
const generateRewards = () => {
    const rewards = []
    const floor = dungeonState.value.floor
    
    // 灵石奖励
    const baseStones = 10 * floor
    rewards.push({
        type: 'spirit_stones',
        amount: baseStones
    })
    
    // 装备奖励（概率获得）
    if (Math.random() < 0.3 || floor % 5 === 0) {
        rewards.push({
            type: 'equipment',
            item: generateEquipment(floor)
        })
    }
    
    return rewards
}
</script>

<template>
    <div class="dungeon-container">
        <n-card title="秘境探索">
            <template #header-extra>
                <n-button
                    type="primary"
                    @click="startDungeon"
                    :disabled="dungeonState.inCombat || dungeonState.showingOptions"
                >
                    开始探索
                </n-button>
            </template>
            
            <n-space vertical>
                <!-- 层数显示 -->
                <n-statistic
                    label="当前层数"
                    :value="dungeonState.floor"
                />
                
                <!-- 选项界面 -->
                <n-card v-if="dungeonState.showingOptions" title="选择增益">
                    <div class="option-cards">
                        <div
                            v-for="option in dungeonState.currentOptions"
                            :key="option.id"
                            class="option-card"
                            :style="{
                                borderColor: getOptionColor(dungeonState.floor)
                            }"
                            @click="selectOption(option)"
                        >
                            <div class="option-name">{{ option.name }}</div>
                            <div class="option-description">{{ option.description }}</div>
                            <div class="option-quality" :style="{ color: getOptionColor(dungeonState.floor) }">
                                {{ dungeonState.floor >= 10 ? '史诗' : dungeonState.floor % 5 === 0 ? '稀有' : '普通' }}
                            </div>
                        </div>
                    </div>
                </n-card>
                
                <!-- 战斗界面 -->
                <template v-if="dungeonState.inCombat && dungeonState.combatManager">
                    <n-card title="战斗信息">
                        <!-- 玩家属性 -->
                        <n-descriptions bordered title="玩家属性">
                            <n-descriptions-item label="生命值">
                                {{ dungeonState.combatManager.player.currentHealth.toFixed(1) }} /
                                {{ dungeonState.combatManager.player.stats.maxHealth }}
                            </n-descriptions-item>
                            <n-descriptions-item label="攻击力">
                                {{ dungeonState.combatManager.player.stats.damage.toFixed(1) }}
                            </n-descriptions-item>
                            <n-descriptions-item label="防御力">
                                {{ dungeonState.combatManager.player.stats.defense.toFixed(1) }}
                            </n-descriptions-item>
                            <n-descriptions-item label="速度">
                                {{ dungeonState.combatManager.player.stats.speed.toFixed(1) }}
                            </n-descriptions-item>
                        </n-descriptions>
                        
                        <n-collapse>
                            <n-collapse-item title="战斗属性">
                                <n-descriptions bordered>
                                    <n-descriptions-item label="暴击率">
                                        {{ (dungeonState.combatManager.player.stats.critRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="连击率">
                                        {{ (dungeonState.combatManager.player.stats.comboRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="反击率">
                                        {{ (dungeonState.combatManager.player.stats.counterRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="眩晕率">
                                        {{ (dungeonState.combatManager.player.stats.stunRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="闪避率">
                                        {{ (dungeonState.combatManager.player.stats.dodgeRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="吸血率">
                                        {{ (dungeonState.combatManager.player.stats.vampireRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                </n-descriptions>
                            </n-collapse-item>
                            
                            <n-collapse-item title="战斗抗性">
                                <n-descriptions bordered>
                                    <n-descriptions-item label="抗暴击">
                                        {{ (dungeonState.combatManager.player.stats.critResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗连击">
                                        {{ (dungeonState.combatManager.player.stats.comboResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗反击">
                                        {{ (dungeonState.combatManager.player.stats.counterResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗眩晕">
                                        {{ (dungeonState.combatManager.player.stats.stunResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗闪避">
                                        {{ (dungeonState.combatManager.player.stats.dodgeResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗吸血">
                                        {{ (dungeonState.combatManager.player.stats.vampireResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                </n-descriptions>
                            </n-collapse-item>
                            
                            <n-collapse-item title="特殊属性">
                                <n-descriptions bordered>
                                    <n-descriptions-item label="强化治疗">
                                        {{ (dungeonState.combatManager.player.stats.healBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="强化爆伤">
                                        {{ (dungeonState.combatManager.player.stats.critDamageBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="弱化爆伤">
                                        {{ (dungeonState.combatManager.player.stats.critDamageReduce * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="最终增伤">
                                        {{ (dungeonState.combatManager.player.stats.finalDamageBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="最终减伤">
                                        {{ (dungeonState.combatManager.player.stats.finalDamageReduce * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="战斗属性提升">
                                        {{ (dungeonState.combatManager.player.stats.combatBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="战斗抗性提升">
                                        {{ (dungeonState.combatManager.player.stats.resistanceBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                </n-descriptions>
                            </n-collapse-item>
                        </n-collapse>
                        
                        <!-- 敌人属性 -->
                        <n-descriptions bordered title="敌人属性" style="margin-top: 16px">
                            <n-descriptions-item label="生命值">
                                {{ dungeonState.combatManager.enemy.currentHealth.toFixed(1) }} /
                                {{ dungeonState.combatManager.enemy.stats.maxHealth }}
                            </n-descriptions-item>
                            <n-descriptions-item label="攻击力">
                                {{ dungeonState.combatManager.enemy.stats.damage.toFixed(1) }}
                            </n-descriptions-item>
                            <n-descriptions-item label="防御力">
                                {{ dungeonState.combatManager.enemy.stats.defense.toFixed(1) }}
                            </n-descriptions-item>
                            <n-descriptions-item label="速度">
                                {{ dungeonState.combatManager.enemy.stats.speed.toFixed(1) }}
                            </n-descriptions-item>
                        </n-descriptions>
                        
                        <n-collapse>
                            <n-collapse-item title="战斗属性">
                                <n-descriptions bordered>
                                    <n-descriptions-item label="暴击率">
                                        {{ (dungeonState.combatManager.enemy.stats.critRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="连击率">
                                        {{ (dungeonState.combatManager.enemy.stats.comboRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="反击率">
                                        {{ (dungeonState.combatManager.enemy.stats.counterRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="眩晕率">
                                        {{ (dungeonState.combatManager.enemy.stats.stunRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="闪避率">
                                        {{ (dungeonState.combatManager.enemy.stats.dodgeRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="吸血率">
                                        {{ (dungeonState.combatManager.enemy.stats.vampireRate * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                </n-descriptions>
                            </n-collapse-item>
                            
                            <n-collapse-item title="战斗抗性">
                                <n-descriptions bordered>
                                    <n-descriptions-item label="抗暴击">
                                        {{ (dungeonState.combatManager.enemy.stats.critResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗连击">
                                        {{ (dungeonState.combatManager.enemy.stats.comboResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗反击">
                                        {{ (dungeonState.combatManager.enemy.stats.counterResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗眩晕">
                                        {{ (dungeonState.combatManager.enemy.stats.stunResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗闪避">
                                        {{ (dungeonState.combatManager.enemy.stats.dodgeResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="抗吸血">
                                        {{ (dungeonState.combatManager.enemy.stats.vampireResist * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                </n-descriptions>
                            </n-collapse-item>
                            
                            <n-collapse-item title="特殊属性">
                                <n-descriptions bordered>
                                    <n-descriptions-item label="强化治疗">
                                        {{ (dungeonState.combatManager.enemy.stats.healBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="强化爆伤">
                                        {{ (dungeonState.combatManager.enemy.stats.critDamageBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="弱化爆伤">
                                        {{ (dungeonState.combatManager.enemy.stats.critDamageReduce * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="最终增伤">
                                        {{ (dungeonState.combatManager.enemy.stats.finalDamageBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="最终减伤">
                                        {{ (dungeonState.combatManager.enemy.stats.finalDamageReduce * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="战斗属性提升">
                                        {{ (dungeonState.combatManager.enemy.stats.combatBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                    <n-descriptions-item label="战斗抗性提升">
                                        {{ (dungeonState.combatManager.enemy.stats.resistanceBoost * 100).toFixed(1) }}%
                                    </n-descriptions-item>
                                </n-descriptions>
                            </n-collapse-item>
                        </n-collapse>
                        
                        <!-- 战斗日志 -->
                        <log-panel
                            ref="logRef"
                            :messages="combatLog"
                            style="margin-top: 16px"
                        />
                    </n-card>
                </template>
            </n-space>
        </n-card>
    </div>
</template>

<style scoped>
.dungeon-container {
    margin: 0 auto;
}

.option-cards {
    display: flex;
    gap: 16px;
    padding: 16px;
    margin: 0 auto;
}

.option-card {
    position: relative;
    padding: 20px;
    border: 2px solid;
    border-radius: 12px;
    background: var(--n-color);
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    min-height: 100px;
    width: 33%;
}

.option-card:hover {
    transform: translateX(5px);
    box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.1);
}

.option-name {
    font-size: 1.3em;
    font-weight: bold;
    margin-bottom: 12px;
    padding-right: 80px;
}

.option-description {
    flex-grow: 1;
    font-size: 1em;
    color: var(--n-text-color);
    line-height: 1.6;
    margin-bottom: 8px;
}

.option-quality {
    position: absolute;
    top: 20px;
    right: 20px;
    font-size: 0.9em;
    font-weight: bold;
    padding: 4px 12px;
    border-radius: 20px;
    background: var(--n-color);
}
</style>