<script setup lang="ts">
import { ref } from 'vue';
import { onMounted } from 'vue';
import PresetModal from './PresetModal.vue';
import type { PresetConfig } from '../types/caddy';
import { presets } from '../presets';

interface Props {
  initialPresetName?: string;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  select: [preset: PresetConfig];
}>();

const showModal = ref(false);
const selectedPreset = ref<PresetConfig | null>(null);

onMounted(() => {
  if (props.initialPresetName) {
    const preset = presets.find(p => p.name === props.initialPresetName);
    if (preset) {
      selectedPreset.value = preset;
    }
  }
});

function handleSelect(preset: PresetConfig) {
  selectedPreset.value = preset;
  emit('select', preset);
}

function clearSelection() {
  selectedPreset.value = null;
}
</script>

<template>
  <div class="space-y-2">
    <div v-if="selectedPreset" class="flex items-center gap-2">
      <div class="flex-1 p-2 bg-muted rounded text-sm">{{ selectedPreset.name }} (Port: {{ selectedPreset.port }})</div>
      <button 
        @click="clearSelection"
        class="p-2 text-destructive hover:text-destructive/90 transition-colors"
      >
        Clear
      </button>
    </div>
    
    <button
      @click="showModal = true"
      type="button"
      class="w-full px-4 py-2 bg-secondary hover:bg-secondary/90 text-secondary-foreground rounded transition-colors"
    >
      {{ selectedPreset ? 'Change Preset' : 'Select from Presets' }}
    </button>
    
    <PresetModal
      :show="showModal"
      @close="showModal = false"
      @select="handleSelect"
    />
  </div>
</template>

<style>
</style>