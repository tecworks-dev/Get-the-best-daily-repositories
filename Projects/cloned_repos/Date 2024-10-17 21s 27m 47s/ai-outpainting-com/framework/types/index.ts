import { SVGProps } from 'react'

export * from './grids'
export * from './products'

export type IconSvgProps = SVGProps<SVGSVGElement> & {
  size?: number
}
