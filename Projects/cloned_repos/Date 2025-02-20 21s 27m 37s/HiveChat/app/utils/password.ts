import bcrypt from "bcryptjs";

export async function verifyPassword(inputPassword: string, storedHashedPassword: string) {
  const isMatch = await bcrypt.compare(inputPassword, storedHashedPassword);
  return isMatch;
}