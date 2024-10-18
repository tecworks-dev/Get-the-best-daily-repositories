import { Select, SelectItem } from '@nextui-org/react'
import type { SelectProps } from '@nextui-org/select'
import React, { forwardRef } from 'react'
import { DictData } from '@prisma/client'
import {omit} from 'radash'
import useDictEntries from '@/framework/hooks/useDictEntries'

export type UeDictSelectProps = {
  dictType: string
  defaultValue?: string | number
} & Partial<SelectProps>

const UeDictSelect = forwardRef<HTMLElement, UeDictSelectProps>(
  (props: UeDictSelectProps, ref) => {
    const { dictType, defaultValue } = props
    const [entries] = useDictEntries(dictType)
    return (
      <Select
        {...omit(props, ['dictType', 'defaultValue', 'ref'])}
        className={`items-center ${props.className}`}
        label={<label className="whitespace-nowrap">{props.label}</label>}
        ref={ref as any}
      >
        {entries.map((it: DictData, index) => (
          <SelectItem value={it.value} key={it.value}>
            {it.label}
          </SelectItem>
        ))}
      </Select>
    )
  },
)
UeDictSelect.displayName = 'UeDictSelect'

export default UeDictSelect
