<template>
  <header class="sticky top-0 z-20 border-secondary-100 bg-white/80 shadow-sm backdrop-blur-lg">
    <div class="container mx-auto flex items-center justify-between py-3">
      <div class="flex items-center justify-between gap-12">
        <NuxtLink
          :to="localePath('/')"
          class="font-display group inline-flex items-center gap-2 text-xl font-semibold tracking-tight text-secondary-700 transition-transform hover:text-blue-500 active:scale-95 md:text-2xl">
          💊
          <div class="flex items-center gap-0.5">进口原研药目录</div>
        </NuxtLink>
      </div>
      <span class="inline-flex items-center gap-2 text-sm text-slate-400">
        Made by
        <a href="https://x.com/lvwzhen" target="_blank" class="transition-all hover:scale-105 hover:text-primary-500">
          lvwzhen
        </a>
      </span>
    </div>
  </header>
</template>

<script>
export default {
  name: "Header",
  data() {
    return {};
  },
};
</script>
