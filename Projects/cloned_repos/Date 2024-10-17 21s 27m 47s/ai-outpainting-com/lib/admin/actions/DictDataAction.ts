'use server'
import service from '@/framework/services/DictDataService'
import { DictData } from '@prisma/client'
import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'


export async function createDictData(prevState: unknown, dictData: DictData) {
  const existDict = await service.getByValue(dictData.type, dictData.value)
  if (existDict != null && !dictData.id) {
    return {
      message: '字典值已存在，不可重复添加'
    }
  }
  if (dictData.id) {
    await service.update(dictData)
  } else {
    await service.create(dictData)
  }
  const path = `/admin/dict-data/${dictData.type}`
  revalidatePath(path)
  redirect(path)
}

export async function deleteDictData(dictData: DictData) {
  if (!dictData.type) {
    throw new Error('字典Type不能为空')
  }
  await service.delete(dictData.id)
  const path = `/admin/dict-data/${dictData.type}`
  revalidatePath(path)
  redirect(path)
}

export async function getByDictType(dictType: string) {

}