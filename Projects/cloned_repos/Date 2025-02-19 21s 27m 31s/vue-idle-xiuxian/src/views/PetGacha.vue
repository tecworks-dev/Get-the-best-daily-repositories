<script setup>
import { usePlayerStore } from '../stores/player'
import { ref } from 'vue'
import { useMessage } from 'naive-ui'
import { InformationCircleOutline } from '@vicons/ionicons5'

const playerStore = usePlayerStore()
const message = useMessage()

// 抽卡动画状态
const isShaking = ref(false)
const isHatching = ref(false)
const showResult = ref(false)
const gachaResult = ref(null)
const showProbabilityInfo = ref(false)
const isDrawing = ref(false)

// 灵宠品质配置
const petRarities = {
    divine: {
        name: '神品',
        color: '#FF0000',
        probability: 0.02,
        essenceBonus: 50
    },
    celestial: {
        name: '仙品',
        color: '#FFD700',
        probability: 0.08,
        essenceBonus: 30
    },
    mystic: {
        name: '玄品',
        color: '#9932CC',
        probability: 0.15,
        essenceBonus: 20
    },
    spiritual: {
        name: '灵品',
        color: '#1E90FF',
        probability: 0.25,
        essenceBonus: 10
    },
    mortal: {
        name: '凡品',
        color: '#32CD32',
        probability: 0.5,
        essenceBonus: 5
    }
}

// 灵宠池配置
const petPool = {
    divine: [
        { name: '玄武', description: '北方守护神兽' },
        { name: '白虎', description: '西方守护神兽' },
        { name: '朱雀', description: '南方守护神兽' },
        { name: '青龙', description: '东方守护神兽' },
        { name: '应龙', description: '上古神龙，掌控风雨' },
        { name: '麒麟', description: '祥瑞之兽，通晓万物' },
        { name: '饕餮', description: '贪婪之兽，吞噬万物，象征无尽的欲望' },
        { name: '穷奇', description: '邪恶之兽，背信弃义，象征混乱与背叛' },
        { name: '梼杌', description: '凶暴之兽，顽固不化，象征无法驯服的野性' },
        { name: '混沌', description: '无序之兽，无形无相，象征原始的混乱' }
    ],
    celestial: [
        { name: '囚牛', description: '龙之长子，喜好音乐，常立于琴头' },
        { name: '睚眦', description: '龙之次子，性格刚烈，嗜杀好斗，常刻于刀剑之上' },
        { name: '嘲风', description: '龙之三子，形似兽，喜好冒险，常立于殿角' },
        { name: '蒲牢', description: '龙之四子，形似龙而小，性好鸣，常铸于钟上' },
        { name: '狻猊', description: '龙之五子，形似狮子，喜静好坐，常立于香炉' },
        { name: '霸下', description: '龙之六子，形似龟，力大无穷，常背负石碑' },
        { name: '狴犴', description: '龙之七子，形似虎，明辨是非，常立于狱门' },
        { name: '负屃', description: '龙之八子，形似龙，雅好诗文，常盘于碑顶' },
        { name: '螭吻', description: '龙之九子，形似鱼，能吞火，常立于屋脊' }
    ],
    mystic: [
        { name: '火凤凰', description: '浴火重生的永恒之鸟' },
        { name: '雷鹰', description: '掌控雷电的猛禽' },
        { name: '冰狼', description: '冰原霸主' },
        { name: '岩龟', description: '坚不可摧的守护者' }
    ],
    spiritual: [
        { name: '玄龟', description: '擅长防御的水系灵宠' },
        { name: '风隼', description: '速度极快的飞行灵宠' },
        { name: '地甲', description: '坚固的大地守护者' },
        { name: '云豹', description: '敏捷的猎手' }
    ],
    mortal: [
        { name: '灵猫', description: '敏捷的小型灵宠' },
        { name: '幻蝶', description: '美丽的蝴蝶灵宠' },
        { name: '火鼠', description: '活泼的啮齿类灵宠' },
        { name: '草兔', description: '温顺的兔类灵宠' }
    ]
}

// 抽取单个灵宠
const drawSinglePet = () => {
    const random = Math.random()
    let accumulatedProb = 0
    
    for (const [rarity, config] of Object.entries(petRarities)) {
        accumulatedProb += config.probability
        if (random <= accumulatedProb) {
            const pool = petPool[rarity]
            const pet = pool[Math.floor(Math.random() * pool.length)]
            // 根据品质生成升级道具数量
            const upgradeItemCount = {
                divine: 5,
                celestial: 4,
                mystic: 3,
                spiritual: 2,
                mortal: 1
            }
            
            return {
                ...pet,
                rarity,
                type: 'pet',
                quality: {
                    strength: Math.floor(Math.random() * 10) + 1,
                    agility: Math.floor(Math.random() * 10) + 1,
                    intelligence: Math.floor(Math.random() * 10) + 1,
                    constitution: Math.floor(Math.random() * 10) + 1
                },
                power: 0,
                experience: 0,
                maxExperience: 100,
                level: 1,
                star: 0,
                upgradeItems: upgradeItemCount[rarity] || 1,
                // 添加战斗属性
                combatAttributes: {
                    attack: (10 + Math.floor(Math.random() * 5)) * (rarity === 'divine' ? 5 : rarity === 'celestial' ? 4 : rarity === 'mystic' ? 3 : rarity === 'spiritual' ? 2 : 1),
                    health: (100 + Math.floor(Math.random() * 20)) * (rarity === 'divine' ? 5 : rarity === 'celestial' ? 4 : rarity === 'mystic' ? 3 : rarity === 'spiritual' ? 2 : 1),
                    defense: (5 + Math.floor(Math.random() * 3)) * (rarity === 'divine' ? 5 : rarity === 'celestial' ? 4 : rarity === 'mystic' ? 3 : rarity === 'spiritual' ? 2 : 1),
                    speed: (10 + Math.floor(Math.random() * 5)) * (rarity === 'divine' ? 3 : rarity === 'celestial' ? 2.5 : rarity === 'mystic' ? 2 : rarity === 'spiritual' ? 1.5 : 1),
                    critRate: Math.random() * 0.1 * (rarity === 'divine' ? 2 : rarity === 'celestial' ? 1.8 : rarity === 'mystic' ? 1.6 : rarity === 'spiritual' ? 1.4 : 1),
                    comboRate: Math.random() * 0.1 * (rarity === 'divine' ? 2 : rarity === 'celestial' ? 1.8 : rarity === 'mystic' ? 1.6 : rarity === 'spiritual' ? 1.4 : 1),
                    counterRate: Math.random() * 0.1 * (rarity === 'divine' ? 2 : rarity === 'celestial' ? 1.8 : rarity === 'mystic' ? 1.6 : rarity === 'spiritual' ? 1.4 : 1),
                    stunRate: Math.random() * 0.1 * (rarity === 'divine' ? 2 : rarity === 'celestial' ? 1.8 : rarity === 'mystic' ? 1.6 : rarity === 'spiritual' ? 1.4 : 1),
                    dodgeRate: Math.random() * 0.1 * (rarity === 'divine' ? 2 : rarity === 'celestial' ? 1.8 : rarity === 'mystic' ? 1.6 : rarity === 'spiritual' ? 1.4 : 1),
                    vampireRate: Math.random() * 0.1 * (rarity === 'divine' ? 2 : rarity === 'celestial' ? 1.8 : rarity === 'mystic' ? 1.6 : rarity === 'spiritual' ? 1.4 : 1)
                }
            }
        }
    }
}

// 执行抽卡
const performGacha = async (times) => {
    const cost = times * 100
    if (playerStore.spiritStones < cost) {
        message.error('灵石不足！')
        return
    }

    if (playerStore.items.filter(item => item.type === 'pet').length >= 100) {
        message.error('灵宠背包已满，请先丢弃一部分灵宠')
        return
    }
    
    if (isDrawing.value) return
    isDrawing.value = true

    // 扣除灵石
    playerStore.spiritStones -= cost

    // 开始抽卡动画
    isShaking.value = true
    await new Promise(resolve => setTimeout(resolve, 1000))
    isShaking.value = false
    
    isHatching.value = true
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 生成抽卡结果
    const results = Array(times).fill().map(() => drawSinglePet())
    
    // 添加到背包并获得精华
    results.forEach(pet => {
        // 添加灵宠到背包
        playerStore.items.push({
            ...pet,
            id: Date.now() + Math.random(),
            level: 1,
            star: 0
        })
        
        // 根据品质获得精华
        const rarityConfig = playerStore.petConfig.rarityMap[pet.rarity]
        if (rarityConfig) {
            playerStore.petEssence += rarityConfig.essenceBonus
        }
    })
    
    // 保存数据
    playerStore.saveData()

    // 显示结果
    gachaResult.value = results
    isHatching.value = false
    showResult.value = true

    // 3秒后隐藏结果
    setTimeout(() => {
        showResult.value = false
        gachaResult.value = null
        isDrawing.value = false
    }, 3000)
}
</script>

<template>
    <n-layout>
        <n-layout-header bordered>
            <n-page-header>
                <template #title>
                    灵宠抽取
                </template>
            </n-page-header>
        </n-layout-header>
        <n-layout-content class="gacha-content">
            <n-card>
                <div class="gacha-container">
                    <div class="spirit-stones">
                        <n-statistic label="灵石" :value="playerStore.spiritStones" />
                    </div>
                    <div class="egg-container">
                        <div class="egg" :class="{
                            'shake': isShaking,
                            'hatch': isHatching
                        }">
                            🥚
                        </div>
                        <div v-if="showResult" class="result-popup">
                            <template v-if="gachaResult.length === 1">
                                <div class="pet-card" :style="{ borderColor: petRarities[gachaResult[0].rarity].color }">
                                    <h3>{{ gachaResult[0].name }}</h3>
                                    <p>品质：{{ petRarities[gachaResult[0].rarity].name }}</p>
                                    <p>{{ gachaResult[0].description }}</p>
                                </div>
                            </template>
                            <template v-else>
                                <div class="multi-result">
                                    <div v-for="pet in gachaResult" :key="pet.name"
                                        class="pet-card-small"
                                        :style="{ borderColor: petRarities[pet.rarity].color }">
                                        <h4>{{ pet.name }}</h4>
                                        <p>{{ petRarities[pet.rarity].name }}</p>
                                    </div>
                                </div>
                            </template>
                        </div>
                    </div>
                    <div class="gacha-buttons">
                        <n-space vertical>
                            <n-space justify="center">
                                <n-button type="primary" @click="performGacha(1)" 
                                    :disabled="playerStore.spiritStones < 100 || isDrawing">
                                    抽取一次 (100灵石)
                                </n-button>
                                <n-button type="primary" @click="performGacha(10)" 
                                    :disabled="playerStore.spiritStones < 1000 || isDrawing">
                                    抽取十次 (1000灵石)
                                </n-button>
                            </n-space>
                            <n-space justify="center">
                                <n-button quaternary circle size="small" @click="showProbabilityInfo = true">
                                    <template #icon>
                                        <n-icon><information-circle-outline /></n-icon>
                                    </template>
                                </n-button>
                            </n-space>
                        </n-space>
                    </div>
                    <!-- 概率说明弹窗 -->
                    <n-modal v-model:show="showProbabilityInfo" preset="card" title="抽卡概率说明">
                        <n-list>
                            <n-list-item v-for="(config, rarity) in petRarities" :key="rarity">
                                <n-space justify="space-between">
                                    <span :style="{ color: config.color }">{{ config.name }}</span>
                                    <span>{{ (config.probability * 100).toFixed(1) }}%</span>
                                </n-space>
                            </n-list-item>
                        </n-list>
                    </n-modal>
                </div>
            </n-card>
        </n-layout-content>
    </n-layout>
</template>

<style scoped>
.gacha-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
}

.spirit-stones {
    align-self: flex-end;
}

.egg-container {
    position: relative;
    width: 200px;
    height: 200px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.egg {
    font-size: 100px;
    transition: transform 0.3s ease;
}

.egg.shake {
    animation: shake 0.5s ease-in-out infinite;
}

.egg.hatch {
    animation: hatch 1s ease-in-out;
}

@keyframes shake {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(-5deg); }
    75% { transform: rotate(5deg); }
}

@keyframes hatch {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.2); opacity: 0.5; }
    100% { transform: scale(0); opacity: 0; }
}

.result-popup {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    animation: popup 0.5s ease-out;
}

@keyframes popup {
    0% { transform: translate(-50%, -50%) scale(0); }
    70% { transform: translate(-50%, -50%) scale(1.1); }
    100% { transform: translate(-50%, -50%) scale(1); }
}

.pet-card {
    background: var(--n-color);
    border: 2px solid;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    min-width: 200px;
}

.multi-result {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
}

.pet-card-small {
    background: var(--n-color);
    border: 2px solid;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
    width: 200px;
}

.gacha-buttons {
    margin-top: 20px;
}

@media screen and (max-width: 768px) {
    .multi-result {
        grid-template-columns: repeat(2, 1fr);
    }
    .pet-card-small {
        width: 170px;
    }
}
</style>