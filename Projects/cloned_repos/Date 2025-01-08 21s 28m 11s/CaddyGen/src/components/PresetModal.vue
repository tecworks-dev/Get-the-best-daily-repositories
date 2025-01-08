<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import { X, Search, Globe, Github } from 'lucide-vue-next';
import { presets } from '../presets';
import type { PresetConfig } from '../types/caddy';
import { nextTick } from 'vue';

function openLink(url: string, event: Event) {
  event.preventDefault();
  event.stopPropagation();
  window.open(url, '_blank', 'noopener,noreferrer');
}

const props = defineProps<{
  show: boolean;
}>();

const emit = defineEmits<{
  close: [];
  select: [preset: PresetConfig];
}>();

const searchQuery = ref('');
const selectedCategory = ref<string>('');
const searchInput = ref<HTMLInputElement | null>(null);

watch(() => props.show, async (newValue) => {
  if (newValue) {
    await nextTick();
    searchInput.value?.focus();
  }
});

const categories = computed(() => ['All', ...new Set(presets.map(p => p.category))]);

const filteredPresets = computed(() => {
  let filtered = presets;
  
  if (selectedCategory.value && selectedCategory.value !== 'All') {
    filtered = filtered.filter(p => p.category === selectedCategory.value);
  }
  
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase();
    filtered = filtered.filter(p => 
      p.name.toLowerCase().includes(query) || 
      p.description.toLowerCase().includes(query)
    );
  }
  
  return filtered;
});

function selectPreset(preset: PresetConfig) {
  emit('select', preset);
  emit('close');
}
</script>

<template>
  <div v-if="show" class="fixed inset-0 z-50 flex items-center justify-center">
    <!-- Backdrop -->
    <div class="absolute inset-0 bg-background/80 backdrop-blur-sm" @click="emit('close')" />
    
    <!-- Modal -->
    <div class="relative bg-card border border-border/50 rounded-lg shadow-lg w-full max-w-2xl max-h-[80vh] overflow-hidden">
      <!-- Header -->
      <div class="flex items-center justify-between p-6 border-b border-border/50">
        <h2 class="text-xl font-semibold">Select Preset</h2>
        <button @click="emit('close')" class="text-muted-foreground hover:text-foreground">
          <X class="w-5 h-5" />
        </button>
      </div>
      
      <!-- Search and Filters -->
      <div class="p-4 border-b border-border/50 space-y-4">
        <div class="relative">
          <Search class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            v-model="searchQuery"
            ref="searchInput"
            type="text"
            placeholder="Search presets..."
            class="w-full pl-10 pr-4 py-2 bg-background border border-border rounded-lg"
          />
        </div>
        
        <div class="flex gap-2 overflow-x-auto pb-2 -mb-2 scrollbar-thin">
          <button
            v-for="category in categories"
            :key="category"
            @click.stop="selectedCategory = category"
            type="button"
            :class="[
              'px-3 py-1.5 rounded-full text-sm whitespace-nowrap flex-shrink-0',
              selectedCategory === category
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
            ]"
          >
            {{ category }}
          </button>
        </div>
      </div>
      
      <!-- Preset List -->
      <div class="overflow-y-auto max-h-[60vh] p-4">
        <div class="grid gap-2">
          <button
            v-for="preset in filteredPresets"
            :key="preset.name"
            @click="selectPreset(preset)"
            class="text-left p-4 rounded-lg border border-border/50 hover:border-primary/50 hover:bg-accent/50 transition-colors"
          >
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-3">
                <div 
                  v-if="preset.logo" 
                  class="w-8 h-8 [&_*]:fill-white flex items-center justify-center overflow-hidden" 
                  style="min-width: 32px;"
                  v-html="preset.logo"
                ></div>
                <strong>{{ preset.name }}</strong>
              </div>
              <div class="flex items-center gap-4">
                <div class="flex gap-2">
                  <a
                    v-if="preset.webLink"
                    :href="preset.webLink"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="text-muted-foreground hover:text-foreground transition-colors"
                    :title="`Visit ${preset.name} website`"
                    @click="openLink(preset.webLink, $event)"
                  >
                    <Globe class="w-4 h-4" />
                  </a>
                  <a
                    v-if="preset.githubLink"
                    :href="preset.githubLink"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="text-muted-foreground hover:text-foreground transition-colors"
                    :title="`View ${preset.name} on GitHub`"
                    @click="openLink(preset.githubLink, $event)"
                  >
                    <Github class="w-4 h-4" />
                  </a>
                </div>
                <span class="text-sm text-muted-foreground">Port: {{ preset.port }}</span>
              </div>
            </div>
            <span class="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary inline-block mt-1">{{ preset.category }}</span>
            <p class="text-sm text-muted-foreground mt-1 pr-4">{{ preset.description }}</p>
          </button>
        </div>
        <div v-if="filteredPresets.length === 0" class="text-center py-8 text-muted-foreground">
          No presets found matching your search
        </div>
      </div>
    </div>
  </div>
</template>