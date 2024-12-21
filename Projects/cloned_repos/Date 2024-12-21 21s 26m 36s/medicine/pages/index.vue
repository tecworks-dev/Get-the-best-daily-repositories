<template>
  <div class="root-container">
    <div>
      <main>
        <div class="container mx-auto">
          <div class="mx-auto my-20 px-4">
            <div class="text-center">
              <h1 class="font-display py-1 text-3xl font-semibold tracking-tight md:text-5xl md:leading-[1.2]">
                进口原研药目录
              </h1>
              <p class="mx-auto mt-4 max-w-3xl text-balance text-secondary-500">
                约 400 种药品，实时更新的进口原研药数据库
              </p>
              <p class="mx-auto mt-2 text-sm text-secondary-500">
                数据来源于网络，仅供参考。用药请遵医嘱，按说明书使用。
              </p>
              <div class="mt-6 flex justify-center">
                <a href="https://github.com/lvwzhen/medicine" class="flex items-center gap-1  bg-secondary-700 px-4 py-2 text-sm font-normal text-white shadow-sm transition-all hover:bg-secondary-800 rounded-full">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="h-6 w-6"><path fill="currentColor" d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5c.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34c-.46-1.16-1.11-1.47-1.11-1.47c-.91-.62.07-.6.07-.6c1 .07 1.53 1.03 1.53 1.03c.87 1.52 2.34 1.07 2.91.83c.09-.65.35-1.09.63-1.34c-2.22-.25-4.55-1.11-4.55-4.92c0-1.11.38-2 1.03-2.71c-.1-.25-.45-1.29.1-2.64c0 0 .84-.27 2.75 1.02c.79-.22 1.65-.33 2.5-.33s1.71.11 2.5.33c1.91-1.29 2.75-1.02 2.75-1.02c.55 1.35.2 2.39.1 2.64c.65.71 1.03 1.6 1.03 2.71c0 3.82-2.34 4.66-4.57 4.91c.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2"/></svg>      贡献纠错
                </a>
              </div>
            </div>
          </div>

          <!-- table -->
          <div class="my-20 text-sm">
            <vue-good-table
              ref="vgtRef"
              :columns="columns"
              :rows="rows"
              :search-options="{
                enabled: true,
                placeholder: '搜索药品名称或厂家'
              }">
              <template slot="table-row" slot-scope="props">
                <span v-if="props.column.field == 'letter'">
                  {{ props.formattedRow[props.column.field] }}
                </span>
                <span v-else-if="props.column.field == 'image'" class="flex justify-center relative">
                  <img 
                    v-if="props.formattedRow.brandName && getImagePath(props.formattedRow.name, props.formattedRow.brandName)"
                    :src="getImagePath(props.formattedRow.name, props.formattedRow.brandName)" 
                    :alt="`${props.formattedRow.name} - ${props.formattedRow.brandName}`"
                    class="h-20 w-20 object-contain rounded-lg border p-1 cursor-pointer"
                    @mouseenter="handleMouseEnter(getImagePath(props.formattedRow.name, props.formattedRow.brandName), $event)"
                    @mouseleave="handleMouseLeave"
                    @error="handleImageError"
                    loading="lazy"
                  />
                </span>
                <span v-else-if="props.column.field == 'manufacturer'" 
                  class="cursor-pointer hover:text-blue-500"
                  @click="handleSearch(props.formattedRow[props.column.field])"
                >
                  {{ props.formattedRow[props.column.field] }}
                </span>
                <span v-else-if="props.column.field == 'tags'" class="flex flex-wrap gap-1">
                  <span 
                    v-for="tag in props.formattedRow[props.column.field]" 
                    :key="tag"
                    class="px-2 py-0.5 text-sm rounded-full bg-blue-100 text-blue-800 cursor-pointer hover:bg-blue-200 transition-colors"
                    @click="handleSearch(tag)"
                  >
                    {{ tag }}
                  </span>
                </span>
                <span v-else>
                  {{ props.formattedRow[props.column.field] }}
                </span>
              </template>
            </vue-good-table>
          </div>
          <!-- table end -->

          <div class="mx-auto my-20 max-w-2xl">
            <h2 class="border-b pb-2 text-2xl font-semibold">Source</h2>
            <ul class="mt-3 list-disc pl-4 leading-relaxed">
              <li>
                <a class="text-blue-500 underline hover:text-purple-700" href="https://mp.weixin.qq.com/s/EBu_ZTy5uovPa_8kCs_TBQ">
                  https://mp.weixin.qq.com/s/EBu_ZTy5uovPa_8kCs_TBQ
                </a>
              </li>
             
            </ul>
          </div>
        </div>
      </main>

      <!-- end  -->
    </div>

    <!-- 添加悬浮图片层 -->
    <div
      v-show="hoveredImage"
      class="fixed z-50 pointer-events-none"
      :style="{
        left: mousePosition.x + 'px',
        top: mousePosition.y + 'px'
      }"
    >
      <img
        :src="hoveredImage"
        class="w-64 h-64 object-contain bg-white rounded-lg shadow-lg border-2 border-gray-200"
      />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import tableData from "~/data/tableData.json";

const hoveredImage = ref('')
const mousePosition = ref({ x: 0, y: 0 })
const searchTerm = ref('')
const vgtRef = ref(null)

const handleSearch = (text) => {
  if (vgtRef.value) {
    vgtRef.value.globalSearchTerm = text
  }
}

const handleMouseEnter = (imageUrl, event) => {
  hoveredImage.value = imageUrl
  updateMousePosition(event)
}

const handleMouseLeave = () => {
  hoveredImage.value = ''
}

const updateMousePosition = (event) => {
  mousePosition.value = {
    x: event.clientX + 20,
    y: event.clientY + 20
  }
}

const handleMouseMove = (event) => {
  if (hoveredImage.value) {
    updateMousePosition(event)
  }
}

const handleImageError = (e) => {
  e.target.src = 'https://dummyimage.com/80x80/fff/aaa';
}
</script>

<script>
export default {

  data() {
    return {
      columns: [
        {
          label: '首字母',
          field: 'letter',
          width: '80px',
          sortable: true,
        },
        {
          label: '药品图片',
          field: 'image',
          width: '120px',
        },
        {
          label: '药品名称',
          field: 'name',
          sortable: true,
        },
        {
          label: '商品名称',
          field: 'brandName',
          sortable: true,
        },
        {
          label: '生产厂家',
          field: 'manufacturer',
          sortable: true,
        },
        {
          label: '标签',
          field: 'tags',
        }
      ],
      rows: tableData,
    };
  },
  methods: {
    getImagePath(name, brandName) {
      if (!brandName) return null
      return `/img/${encodeURIComponent(name)}-${encodeURIComponent(brandName)}.jpg`
    },
    handleSearch(text) {
      if (this.vgtRef) {
        this.vgtRef.globalSearchTerm = text
      }
    },
  }
}
</script>

<style>
.vgt-table {
  @apply border-spacing-0 border-separate;
}
.vgt-table td {
  @apply border-b border-gray-200 p-4;
}
.vgt-table th {
  @apply bg-gray-50 border-b border-gray-200 p-4 font-medium text-gray-900;
}
.vgt-input {
  @apply w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-primary-500 focus:outline-none;
}
.vgt-input,
.vgt-select {
  border: 1px solid #eee;
}
tr:hover {
  background: linear-gradient(#f4f5f8, #f1f3f6);
  transition: all 0.3s;
}
table.vgt-table {
  font-size: 14px;
}
table.vgt-table td {
  vertical-align: middle;
}
</style>
