<script setup lang="ts">
import { ref } from 'vue';
import { X, Upload } from 'lucide-vue-next';
import type { CaddyHost } from '../types/caddy';

const props = defineProps<{
  show: boolean;
}>();

const emit = defineEmits<{
  close: [];
  import: [hosts: CaddyHost[]];
}>();

const caddyfileContent = ref('');

function parseCaddyfile(content: string): CaddyHost[] {
  const hosts: CaddyHost[] = [];
  // First, clean up the content and split into blocks
  const blocks = content
    .split(/\n(?=\S)/)
    .reduce((acc: string[], line) => {
      if (line.includes('{')) {
        acc.push(line);
      } else if (line.trim() && !line.trim().startsWith('#')) {
        if (acc.length > 0) {
          acc[acc.length - 1] += '\n' + line;
        }
      }
      return acc;
    }, [])
    .map(block => block.trim())
    .filter(block => {
      // Keep blocks that aren't empty and aren't just comments
      const nonCommentLines = block.split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'));
      return nonCommentLines.length > 0;
    });
  
  for (const block of blocks) {
    const lines = block.split('\n')
      .map(line => line.trim())
      .filter(line => line && !line.startsWith('#'));
    if (lines.length === 0) continue;
    
    const domainLine = lines[0].trim();
    if (!domainLine || domainLine === '}') continue;
    
    let domain = domainLine.split(' ')[0];
    domain = domain === ':80' ? ':80' : domain;
    
    const host: CaddyHost = {
      id: crypto.randomUUID(),
      domain,
      gzip: false
    };

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line === '{' || line === '}') continue;

      if (line.startsWith('root *')) {
        const root = line.split('root *')[1].trim();
        host.fileServer = {
          root,
          browse: false,
          php: false,
          hide: []
        };
      } else if (line.startsWith('file_server')) {
        if (!host.fileServer) {
          host.fileServer = {
            root: '/',
            browse: line.includes('browse'),
            php: false,
            hide: []
          };
        } else {
          host.fileServer.browse = line.includes('browse');
        }
      } else if (line.startsWith('reverse_proxy')) {
        host.reverseProxy = line.split(' ').slice(1).join(' ');
      } else if (line.startsWith('encode gzip')) {
        host.gzip = true;
      } else if (line.startsWith('tls')) {
        const email = line.split(' ')[1];
        host.tls = {
          email: email === 'internal' ? undefined : email,
          selfSigned: email === 'internal'
        };
      }
    }
    
    hosts.push(host);
  }
  
  return hosts;
}

function handleFileUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    caddyfileContent.value = e.target?.result as string;
  };
  reader.readAsText(file);
}

function importCaddyfile() {
  const hosts = parseCaddyfile(caddyfileContent.value);
  emit('import', hosts);
  emit('close');
  caddyfileContent.value = '';
}
</script>

<template>
  <div v-if="show" class="fixed inset-0 z-50 flex items-center justify-center">
    <!-- Backdrop -->
    <div class="absolute inset-0 bg-background/80 backdrop-blur-sm" @click="emit('close')" />
    
    <!-- Modal -->
    <div class="relative bg-card border border-border/50 rounded-lg shadow-lg w-full max-w-2xl">
      <!-- Header -->
      <div class="flex items-center justify-between p-6 border-b border-border/50">
        <h2 class="text-xl font-semibold">Import Caddyfile</h2>
        <button @click="emit('close')" class="text-muted-foreground hover:text-foreground">
          <X class="w-5 h-5" />
        </button>
      </div>
      
      <!-- Content -->
      <div class="p-6 space-y-4">
        <div>
          <label class="block mb-2">Paste your Caddyfile content:</label>
          <textarea
            v-model="caddyfileContent"
            class="w-full h-48 p-4 bg-muted rounded-lg font-mono text-sm"
            placeholder="example.com {
    reverse_proxy localhost:3000
}"
          ></textarea>
        </div>
        
        <div class="text-center">
          <span class="text-muted-foreground">or</span>
        </div>
        
        <div class="flex items-center justify-center">
          <label class="relative cursor-pointer bg-secondary hover:bg-secondary/90 text-secondary-foreground rounded-lg px-4 py-2 transition-colors">
            <input
              type="file"
              accept="*"
              class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              @change="handleFileUpload"
            />
            <div class="flex items-center gap-2">
              <Upload class="w-4 h-4" />
              Upload Caddyfile
            </div>
          </label>
        </div>
      </div>
      
      <!-- Footer -->
      <div class="flex justify-end gap-2 p-6 border-t border-border/50 bg-muted/50">
        <button
          @click="emit('close')"
          class="px-4 py-2 bg-secondary hover:bg-secondary/90 text-secondary-foreground rounded-lg transition-colors"
        >
          Cancel
        </button>
        <button
          @click="importCaddyfile"
          class="px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg transition-colors"
          :disabled="!caddyfileContent"
        >
          Import
        </button>
      </div>
    </div>
  </div>
</template>