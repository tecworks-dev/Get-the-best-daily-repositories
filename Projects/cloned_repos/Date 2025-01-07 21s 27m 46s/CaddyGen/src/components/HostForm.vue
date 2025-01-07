<script setup lang="ts">
import { ref } from 'vue';
import type { CaddyHost } from '../types/caddy';
import PresetSelect from './PresetSelect.vue';
import type { PresetConfig } from '../types/caddy';

const props = defineProps<{
  initialHost?: CaddyHost;
}>();

const emit = defineEmits<{
  save: [host: CaddyHost];
  cancel: [];
}>();

const host = ref<CaddyHost>(props.initialHost || {
  id: crypto.randomUUID(),
  domain: '',
  fileServer: {
    root: '',
    browse: false,
    php: false,
    hide: []
  },
  gzip: false,
  tls: {
    email: '',
    selfSigned: false
  },
  security: {
    ipFilter: {
      enabled: false,
      allow: [],
      block: []
    },
    rateLimit: {
      enabled: false,
      requests: 100,
      window: '1m'
    },
    cspEnabled: false,
    csp: '',
    forwardAuth: {
      enabled: false,
      url: '',
      verifyHeader: '',
      verifyValue: ''
    }
  },
  cors: {
    enabled: false,
    allowOrigins: [],
    allowMethods: [],
    allowHeaders: []
  },
  performance: {
    brotli: false,
    cacheControlEnabled: false,
    cacheControl: ''
  }
});

const serverType = ref(host.value.fileServer ? 'fileServer' : '');

function handleServerTypeChange(event: Event) {
  const value = (event.target as HTMLSelectElement).value;
  serverType.value = value;
  if (serverType.value === 'fileServer') {
    host.value.reverseProxy = undefined;
    if (!host.value.fileServer) {
      host.value.fileServer = {
        root: '/var/www/html',
        browse: false,
        php: false,
        hide: []
      };
    }
  } else {
    host.value.fileServer = undefined;
  }
}
const showAdvanced = ref(false);

function handleSubmit() {
  emit('save', host.value);
}

function applyPreset(preset: PresetConfig) {
  host.value.reverseProxy = `http://localhost:${preset.port}`;
  host.value.presetName = preset.name;
  host.value.fileServer = undefined;
  serverType.value = '';
}
</script>

<template>
  <form @submit.prevent="handleSubmit" class="host-form bg-card rounded-lg border border-border/50 shadow-lg">
    <div class="p-6">
      <h2 class="text-2xl font-semibold mb-6 text-foreground">
        {{ initialHost ? 'Edit' : 'Add New' }} Host
      </h2>

      <div class="space-y-4">
        <div class="form-group">
          <label>Domain:</label>
          <input v-model="host.domain" required placeholder="example.com" />
        </div>
    
        <div class="form-group">
          <label>Preset (optional):</label>
          <PresetSelect 
            :initial-preset-name="host.presetName"
            @select="applyPreset" 
          />
        </div>

        <div class="form-group">
          <label>Type:</label>
          <select v-model="serverType" @change="handleServerTypeChange">
            <option value="">Reverse Proxy</option>
            <option value="fileServer">File Server</option>
          </select>
        </div>

        <template v-if="serverType === 'fileServer'">
          <div class="form-group">
            <label>Root Directory:</label>
            <input v-model="host.fileServer.root" placeholder="/var/www/html" />
          </div>
          <div class="form-group">
            <label class="checkbox">
              <input type="checkbox" v-model="host.fileServer.browse" />
              Enable directory listing
            </label>
          </div>
          <div class="form-group">
            <label class="checkbox">
              <input type="checkbox" v-model="host.fileServer.php" />
              Enable PHP support
            </label>
          </div>
        </template>

        <template v-if="serverType === ''">
          <div class="form-group">
            <label>Proxy To:</label>
            <input
              v-model="host.reverseProxy"
              placeholder="http://localhost:3000"
              required
            />
          </div>
        </template>

        <div class="form-group">
          <label class="checkbox">
            <input type="checkbox" v-model="host.gzip" />
            Enable Gzip compression
          </label>
        </div>

        <button type="button" @click="showAdvanced = !showAdvanced" class="btn-link">
          {{ showAdvanced ? 'Hide' : 'Show' }} Advanced Options
        </button>

        <div v-if="showAdvanced" class="advanced-options">
          <div class="form-group">
            <label>TLS Email (for Let's Encrypt):</label>
            <input
              v-model="host.tls.email"
              type="email"
              placeholder="admin@example.com"
            />
          </div>

          <div class="form-group">
            <label class="checkbox">
              <input type="checkbox" v-model="host.tls.selfSigned" />
              Use self-signed certificate
            </label>
          </div>

          <!-- Security Section -->
          <div class="advanced-section">
            <h3 class="text-lg font-semibold mb-4">Security</h3>
            
            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.security.cspEnabled" />
                Enable Content Security Policy
              </label>
            </div>
            
            <div class="form-group" v-if="host.security.cspEnabled">
              <label>Content Security Policy:</label>
              <input
                v-model="host.security.csp"
                placeholder="default-src 'self';"
              />
            </div>

            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.security.ipFilter.enabled" />
                Enable IP Filtering
              </label>
            </div>

            <div v-if="host.security.ipFilter.enabled">
              <label>IP Allow List (one per line):</label>
              <textarea
                :value="host.security.ipFilter.allow.join('\n')"
                @input="e => host.security.ipFilter.allow = (e.target as HTMLTextAreaElement).value.split('\n').filter(Boolean)"
                placeholder="192.168.1.0/24"
                rows="3"
              ></textarea>
              
              <label>IP Block List (one per line):</label>
              <textarea
                :value="host.security.ipFilter.block.join('\n')"
                @input="e => host.security.ipFilter.block = (e.target as HTMLTextAreaElement).value.split('\n').filter(Boolean)"
                placeholder="10.0.0.0/8"
                rows="3"
              ></textarea>
            </div>

            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.security.forwardAuth.enabled" />
                Enable Forward Authentication
              </label>
            </div>

            <div v-if="host.security.forwardAuth.enabled" class="space-y-4" style="margin-bottom:20px;">
              <div class="form-group">
                <label>Authentication URL:</label>
                <input
                  v-model="host.security.forwardAuth.url"
                  placeholder="http://authelia:9091/api/verify"
                  required
                />
              </div>
              <div class="form-group">
                <label>Verify Header (optional):</label>
                <input
                  v-model="host.security.forwardAuth.verifyHeader"
                  placeholder="Remote-User"
                />
              </div>
              <div class="form-group">
                <label>Verify Value (optional):</label>
                <input
                  v-model="host.security.forwardAuth.verifyValue"
                  placeholder="{{user}}"
                />
              </div>
            </div>

            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.security.rateLimit.enabled" />
                Enable Rate Limiting
              </label>
            </div>

            <div class="form-group" v-if="host.security.rateLimit.enabled">
              <label>Rate Limit:</label>
              <div class="flex gap-2">
                <input
                  type="number"
                  v-model="host.security.rateLimit.requests"
                  class="w-24"
                  min="1"
                />
                <span class="text-muted-foreground self-center">requests per</span>
                <input
                  v-model="host.security.rateLimit.window"
                  placeholder="1m"
                  class="w-24"
                />
              </div>
            </div>
          </div>

          <!-- CORS Section -->
          <div class="advanced-section">
            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.cors.enabled" />
                Enable CORS
              </label>
            </div>

            <template v-if="host.cors.enabled">
              <div class="form-group">
                <label>Allow Origins (one per line):</label>
                <textarea
                  :value="host.cors.allowOrigins.join('\n')"
                  @input="e => host.cors.allowOrigins = (e.target as HTMLTextAreaElement).value.split('\n').filter(Boolean)"
                  placeholder="https://example.com"
                  rows="3"
                ></textarea>
              </div>
            </template>
          </div>

          <!-- Performance Section -->
          <div class="advanced-section">
            <h3 class="text-lg font-semibold mb-4">Performance</h3>
            
            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.performance.brotli" />
                Enable Brotli compression
              </label>
            </div>

            <div class="form-group">
              <label class="checkbox">
                <input type="checkbox" v-model="host.performance.cacheControlEnabled" />
                Enable Cache Control
              </label>
            </div>

            <div class="form-group" v-if="host.performance.cacheControlEnabled">
              <label>Cache Control:</label>
              <input
                v-model="host.performance.cacheControl"
                placeholder="public, max-age=3600"
              />
            </div>
          </div>

          <!-- File Server Hide Patterns -->
          <template v-if="serverType === 'fileServer'">
            <div class="advanced-section">
              <h3 class="text-lg font-semibold mb-4">Hidden Files</h3>
              <div class="form-group">
                <label>Hide Patterns (one per line):</label>
                <textarea
                  :value="host.fileServer.hide.join('\n')"
                  @input="e => host.fileServer.hide = (e.target as HTMLTextAreaElement).value.split('\n').filter(Boolean)"
                  placeholder=".git/*
.env
*.log"
                  rows="3"
                ></textarea>
              </div>
            </div>
          </template>
        </div>
      </div>
    </div>

    <div class="form-actions border-t border-border/50 p-6 bg-muted/50">
      <button type="submit" class="px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg transition-colors">
        Save Host
      </button>
      <button 
        type="button" 
        @click="emit('cancel')" 
        class="px-4 py-2 bg-secondary hover:bg-secondary/90 text-secondary-foreground rounded-lg transition-colors"
      >
        Cancel
      </button>
    </div>
  </form>
</template>

<style scoped>
.host-form {
  max-width: 600px;
  margin: 2rem auto;
}

.form-group {
  margin-bottom: 1rem;
  position:relative;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: hsl(var(--foreground));
  font-weight: 500;
}

.form-group label.checkbox {
  padding-left:40px;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid hsl(var(--border));
  border-radius: 4px;
  font-size: 1rem;
  background-color: hsl(var(--background));
  color: hsl(var(--foreground));
}

.checkbox {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  color: hsl(var(--foreground));
  cursor: pointer;
  user-select: none;
  font-size: 0.9375rem;
}

.checkbox input {
  appearance: none;
  -webkit-appearance: none;
  width: 0.875rem;
  height: 0.875rem;
  border: 1.5px solid hsl(var(--border));
  border-radius: 4px;
  background-color: hsl(var(--background));
  cursor: pointer;
  position: relative;
  transition: all 0.2s ease;
  flex-shrink: 0;
  margin: 0;
  position:absolute;
  top: -1px;
  left: 0px;
}

.checkbox input:checked {
  background-color: hsl(var(--primary));
  border-color: hsl(var(--primary));
}

.checkbox input:checked::after {
  content: '';
  position: absolute;
  left: 50%;
  top: 45%;
  width: 4px;
  height: 7px;
  border: solid white;
  border-width: 0 2px 2px 0;
  transform: translate(-50%, -50%) rotate(45deg);
}

.checkbox input:focus-visible {
  outline: 2px solid hsl(var(--ring));
  outline-offset: 2px;
  border-color: hsl(var(--primary));
}

.checkbox:hover input {
  border-color: hsl(var(--primary));
}

.advanced-options {
  margin-top: 1rem;
  padding: 1.5rem;
  border: 1px solid hsl(var(--border));
  border-radius: 4px;
  background-color: hsl(var(--muted));
}

.advanced-section {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid hsl(var(--border));
}

textarea {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid hsl(var(--border));
  border-radius: 4px;
  font-size: 1rem;
  background-color: hsl(var(--background));
  color: hsl(var(--foreground));
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.form-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

button.btn-link {
  background: none;
  border: none;
  color: hsl(var(--muted-foreground));
  padding: 0.5rem 1rem;
  font: inherit;
  cursor: pointer;
  text-decoration: none;
  border-radius: 4px;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

button.btn-link:hover {
  color: hsl(var(--foreground));
  background: hsl(var(--muted));
}
</style>