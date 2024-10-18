/**
 * 通用分页
 */
export interface Pageable {
  /**
   * 当前页
   */
  page?: number | string | null
  /**
   * 页面大小
   */
  size?: number | string | null
}

/**
 * 通用查询
 */
export interface Queryable<T> {
  /**
   * 记录数据
   */
  data?: T[]
  /**
   * 记录条数
   */
  count?: number
}

/**
 * 负责分页查询
 */
export type PageQueryable<T> = { [K in keyof Partial<T>]: any } & Queryable<T> &
  Pageable &
  Record<string, any>

/**
 * 通用表单动作
 */
export interface FormActionable {
  action: 'create' | 'edit' | 'readonly'
  id?: number
}

/**
 * 表单创建
 */
export type FormCreateAction<T> = { [T in keyof Partial<T>]: any } & Pick<
  FormActionable,
  'action'
> &
  Record<string, any>

/**
 * 表单修改
 */
export type FormEditAction<T> = {
  [T in keyof Partial<T>]: any
} & Required<FormActionable> &
  Record<string, any>

/**
 * 只读表单
 */
export type FormReadonlyAction<T> = {
  [T in keyof Readonly<Partial<T>>]: any
} & Required<FormActionable> &
  Record<string, any>
