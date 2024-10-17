import { Pageable } from '../types'
import { omit, pick } from 'radash'

const ITEMS_PER_PAGE = 10

export function offsetPage({ page, size }: Pageable) {
  let offsetPage = page ? Number(page) || 1 : 1
  let offsetSize = size ? Number(size) || ITEMS_PER_PAGE : 10
  const v = (offsetPage - 1) * offsetSize
  return {
    skip: v,
    take: offsetSize ?? ITEMS_PER_PAGE
  }
}

export function stripPage<T extends Pageable>(
  data: Pageable
): { page: Pageable; params: Omit<T, 'page' | 'size'> } {
  const page = pick(data, ['page', 'size']) as Pageable
  const params = omit(data, ['page', 'size']) as T
  return { page, params }
}

export function isEmptyPage(count: number) {
  return count <= 0
}

export function hasMorePage(
  count: number,
  page: number,
  size: number = 10
): boolean {
  return page < Math.floor((count ?? 0) / size)
}

export function getTotalPages(totalCount: number, pageSize: number): number {
  if (!(totalCount || pageSize)) return 0
  return Math.ceil(totalCount / pageSize)
}