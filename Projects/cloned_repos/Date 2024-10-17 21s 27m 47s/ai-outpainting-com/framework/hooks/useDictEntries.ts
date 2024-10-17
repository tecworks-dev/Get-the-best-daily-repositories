import { useEffect, useState } from 'react'
import { DictData } from '@prisma/client'

export default function useDictEntries(
  dictType: string,
): [DictData[], (value: string) => DictData | undefined | null] {
  const [entries, setEntries] = useState<DictData[]>([])
  useEffect(() => {
    async function loadEntries() {
      try {
        // 假设返回的数据是直接可用的选项数组
        const response = await fetch(`/api/dict-data?dictType=${dictType}`)
        return await response.json()
      } catch (error) {
        console.error('Failed to fetch data:', error)
        return Promise.resolve([])
      }
    }

    loadEntries().then((res) => {
      setEntries(res)
    })
  }, [dictType])

  function getEntry(value: string): DictData | undefined | null {
    if (entries.length <= 0) {
      return null
    }
    return entries.find((it) => it.value === value)
  }

  return [entries, getEntry]
}
