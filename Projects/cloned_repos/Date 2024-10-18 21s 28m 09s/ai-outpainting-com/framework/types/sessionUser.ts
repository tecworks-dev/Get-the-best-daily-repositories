import { AdapterUser } from '@auth/core/adapters'

export interface SessionUser extends AdapterUser{
   credit:number
}