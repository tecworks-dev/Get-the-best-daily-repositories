import bcrypt from 'bcryptjs'

export function useHashPassword(pwd: string) {
  return bcrypt.hashSync(pwd, bcrypt.genSaltSync(6))
}
