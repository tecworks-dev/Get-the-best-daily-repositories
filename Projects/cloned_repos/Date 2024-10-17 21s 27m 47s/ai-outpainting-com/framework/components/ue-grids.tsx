'use client'

import React, { memo, Suspense, useEffect, useState } from 'react'
import {
  Button,
  ButtonProps,
  Chip,
  ChipProps,
  Link,
  LinkProps,
  Pagination,
  Select,
  SelectItem,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableColumn,
  TableHeader,
  TableRow,
} from '@nextui-org/react'
import { assign, isFunction, pick } from 'radash'
import { isBoolean } from '../utils'
import {
  IoAddOutline,
  IoPencilOutline,
  IoRefreshOutline,
  IoReloadOutline,
  IoSearchOutline,
  IoTrashOutline,
} from 'react-icons/io5'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { ColumnSize, ColumnStaticSize } from '@react-types/table'
import { Card } from '@nextui-org/card'
import useDictEntries from '@/framework/hooks/useDictEntries'

export interface UePageGridProps extends UePageTableProps {
  query?: UePageQueryProps
  forms?: React.ReactNode
  toolbar?: UePageToolbarProps
  tools?: React.ReactNode
  pagination: UePaginationProps
  children?: React.ReactNode
}

export interface UePageQueryProps {
  onQuery?: (forms: Record<string, any>) => void
  children?: React.ReactNode
}

export interface UePageToolbarProps {
  refreshable?: boolean
  children?: React.ReactNode
}

export interface UePageTableProps {
  layout?: 'auto' | 'fixed'
  isStriped?: boolean
  isCompact?: boolean
  align?: 'start' | 'center' | 'end'
  data?: any[]
  columns: UePageTableColumn[]
  topContent?: React.ReactNode
  bottomContent?: React.ReactNode
  children?: React.ReactNode
}

export interface UePageTableColumn {
  type?: 'seq'
  key: string
  label?: string
  title: string
  align?: 'start' | 'center' | 'end'
  width?: ColumnSize
  minWidth?: ColumnStaticSize
  maxWidth?: ColumnStaticSize
  formatter?:
    | string
    | ((params: {
        cellValue: string | unknown
        row: any
        col: UePageTableColumn
      }) => string)
  render?: (params: {
    cellValue: string | unknown
    row: any
    col: UePageTableColumn
  }) => React.ReactNode
}

export interface UePaginationProps {
  totalCount?: number
  sizePages?: number[]
}

export function AddButton({ href }: { href: string }) {
  return (
    <Button
      as={Link}
      startContent={<IoAddOutline />}
      size="sm"
      color="primary"
      href={href}
    >
      新增
    </Button>
  )
}

export function EditButton({ href }: { href: string }) {
  return (
    <Button
      as={Link}
      size="sm"
      color="primary"
      variant="flat"
      href={href}
      startContent={<IoPencilOutline />}
    >
      编辑
    </Button>
  )
}

export interface DeleteButtonProps {
  message?: string
  onSubmit: () => void
}

export function DeleteButton({ onSubmit, message }: DeleteButtonProps) {
  async function handleSubmit() {
    const confirmed = window.confirm(message ?? '请确实是否删除该记录?')
    if (confirmed) {
      return onSubmit && onSubmit()
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <Button
        size="sm"
        color="default"
        className="border-primary-100"
        variant="bordered"
        type="submit"
        startContent={<IoTrashOutline />}
      >
        删除
      </Button>
    </form>
  )
}

export interface OperateButtonProps extends Partial<ButtonProps> {
  content: string
}

export function OperateButton(props: OperateButtonProps) {
  return (
    <Button as={Link} color="primary" size="sm" variant="flat" {...props}>
      {props.content}
    </Button>
  )
}

export interface ChipCellProps extends Partial<ChipProps> {
  content?: string
}

export function ChipCell(props: ChipCellProps) {
  const ChipMemoize = memo((props: ChipCellProps) => (
    <Chip size="sm" variant="flat" color="primary" {...props}>
      {props.content ?? ''}
    </Chip>
  ))
  ChipMemoize.displayName = 'ChipMemoize'
  return <ChipMemoize {...props}></ChipMemoize>
}

export interface LinkCellProps extends Partial<Omit<LinkProps, 'href'>> {
  href: string
  content?: string
}

export function LinkCell(props: LinkCellProps) {
  const LinkMemoize = memo((props: LinkCellProps) => (
    <Link color="primary" {...props} href={props.href}>
      {props.content ?? props.href}
    </Link>
  ))
  LinkMemoize.displayName = 'LinkMemoize'
  return <LinkMemoize {...props}></LinkMemoize>
}

export interface FileLinkCellProps extends LinkCellProps {}

export interface DictCellProps extends ChipProps {
  dictType: string
  content: string
}

export function DictCell(props: DictCellProps) {
  const ChipCellMemoize = memo((props: Omit<DictCellProps, 'dictType'>) => (
    <ChipCell {...props}></ChipCell>
  ))
  ChipCellMemoize.displayName = 'ChipCellMemoize'

  const [$, getEntry] = useDictEntries(props.dictType)

  const [textValue, setTextValue] = useState<string>('')
  useEffect(() => {
    const it = getEntry(props.content)
    if (it && it.label) {
      setTextValue(it.label ?? props.content)
    }
  }, [$.length, getEntry, props.content])

  return <ChipCellMemoize {...props} content={textValue} />
}

export function UePageQuery({ children, onQuery }: UePageQueryProps) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const router = useRouter()

  function handleSubmit(e: any) {
    e.preventDefault()
    const params = new URLSearchParams(searchParams)
    const data = new FormData(e.target as HTMLFormElement)
    const queryData = {} as Record<string, any>
    Array.from(data.keys()).forEach((k) => {
      const v = data.get(k)
      queryData[k] = v
      if (v) {
        params.set(k, v.toString())
      }
    })
    onQuery && onQuery(queryData)
    router.push(`${pathname}?${params.toString()}`)
  }

  function handleReset(e: any) {
    const params = new URLSearchParams(searchParams)
    const data = new FormData(e.target as HTMLFormElement)
    Array.from(data.keys()).forEach((k) => {
      params.delete(k)
    })
    router.push(`${pathname}?${params.toString()}`)
  }

  return (
    <form onSubmit={handleSubmit} onReset={handleReset}>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {children}
        <div className="flex items-center gap-4">
          <Button
            startContent={<IoSearchOutline />}
            type="submit"
            size="sm"
            color="primary"
            variant="flat"
          >
            查询
          </Button>
          <Button
            startContent={<IoReloadOutline />}
            type="reset"
            size="sm"
            variant="flat"
          >
            重置
          </Button>
        </div>
      </div>
    </form>
  )
}

export function UePageToolbar({ children, refreshable }: UePageToolbarProps) {
  const router = useRouter()

  function onRefresh() {
    router.refresh()
  }

  refreshable = isBoolean(refreshable) ? refreshable : true
  let className = 'my-4 flex items-center justify-between'
  // 如果没有新增按钮，刷新按钮放到最后
  if (!children) {
    className = 'my-4 flex items-center justify-end'
  }
  return (
    <div className={className}>
      {children}
      {refreshable && (
        <Button isIconOnly size="sm" variant="flat" onClick={onRefresh}>
          <IoRefreshOutline />
        </Button>
      )}
    </div>
  )
}

export function UePagination({ totalCount, sizePages }: UePaginationProps) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const router = useRouter()

  const currentPage = Number(searchParams.get('page')) || 1
  const currentSize = Number(searchParams.get('size')) || 10

  const [pageSize, setPageSize] = useState<number>(currentSize)

  function onSizeChange(value: any) {
    const [val] = value
    setPageSize(Number(val))
    const params = new URLSearchParams(searchParams)
    params.set('page', currentPage.toString())
    console.log('v', value)
    params.set('size', val)
    router.push(`${pathname}?${params.toString()}`)
  }

  function onPageChange(pageNum: number) {
    const params = new URLSearchParams(searchParams)
    params.set('page', pageNum.toString())
    params.set('size', pageSize.toString())
    router.push(`${pathname}?${params.toString()}`)
  }

  const pages = sizePages ? sizePages : [5, 10, 15, 20, 25]
  // 根据总条数和分页大小计算总分页数
  const total = Math.ceil((totalCount ?? 0) / pageSize)
  return (
    <div className="mt-4 flex justify-center md:justify-between">
      <div className="flex items-center gap-2">
        <div className="text-default-400 text-small">总条数：{totalCount}</div>
        <div className="flex items-center flex-shrink-0">
          <label className="flex-shrink-0 text-default-400 text-small">
            每页大小：
          </label>
          <Select
            className="flex-shrink-0"
            labelPlacement="outside"
            size="sm"
            selectedKeys={[pageSize.toString()]}
            onSelectionChange={onSizeChange}
          >
            {pages.map((it) => (
              <SelectItem
                classNames={{ base: 'text-default-400' }}
                key={it}
                value={it}
                textValue={it.toString()}
              >
                <span>{it}</span>
              </SelectItem>
            ))}
          </Select>
        </div>
      </div>
      {total > 0 && (
        <Pagination
          total={total}
          page={currentPage}
          size="sm"
          showControls={true}
          onChange={onPageChange}
        ></Pagination>
      )}
    </div>
  )
}

function renderTableCell({
  row,
  rowIndex,
  col,
  colIndex,
}: {
  row: any
  rowIndex: number
  col: UePageTableColumn
  colIndex: number
}) {
  const { type, key, formatter, render } = col
  if ('seq' === type) {
    return <TableCell key={`${rowIndex}_${colIndex}`}>{rowIndex + 1}</TableCell>
  }
  let cellValue = row && key ? row[key] : ''
  const params = { cellValue, row, col }
  if (render && isFunction(render)) {
    return (
      <TableCell key={`${rowIndex}_${colIndex}`}>{render(params)}</TableCell>
    )
  }
  if (formatter && isFunction(formatter)) {
    cellValue = formatter(params)
  }
  return <TableCell key={`${rowIndex}_${colIndex}`}>{cellValue}</TableCell>
}

export function UePageTable(props: UePageTableProps) {
  const { data, columns, topContent, bottomContent } = props
  const tableProps = { isStriped: true }
  const newTableProps = pick(props, ['layout', 'isStriped', 'isCompact'])
  // @ts-ignore
  assign(tableProps, newTableProps)

  return (
    <Table
      aria-label="a page table"
      radius="sm"
      shadow="sm"
      selectionMode="single"
      selectionBehavior="toggle"
      {...tableProps}
      bottomContent={bottomContent}
    >
      <TableHeader>
        {columns.map((it) => (
          <TableColumn
            {...pick(it, ['align', 'width', 'minWidth', 'maxWidth'])}
            key={it.key}
          >
            {it.title ?? it.label}
          </TableColumn>
        ))}
      </TableHeader>
      <TableBody emptyContent={'暂无有效数据'}>
        {(data ?? []).map((row, rowIndex) => (
          <TableRow key={rowIndex}>
            {columns.map((col, colIndex) =>
              renderTableCell({ row, rowIndex, col, colIndex }),
            )}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function UePageTableSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      <Card className="w- space-y-8 p-4" radius="lg">
        <Skeleton className="rounded-lg">
          <div className="h-10 rounded-lg bg-secondary"></div>
        </Skeleton>
        <div className="space-y-2">
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-50"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-300"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-50"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-300"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-50"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-300"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-50"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-300"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-50"></div>
          </Skeleton>
          <Skeleton className="w-full rounded">
            <div className="h-12 w-full rounded bg-secondary-300"></div>
          </Skeleton>
        </div>
      </Card>
    </div>
  )
}

export function UePageGrid(props: UePageGridProps) {
  const { forms, tools, pagination } = props
  return (
    <div>
      <UePageQuery>{forms}</UePageQuery>
      <UePageToolbar>{tools}</UePageToolbar>
      <Suspense fallback={<UePageTableSkeleton />}>
        <UePageTable
          {...props}
          bottomContent={<UePagination {...pagination} />}
        />
      </Suspense>
    </div>
  )
}

export default UePageGrid
