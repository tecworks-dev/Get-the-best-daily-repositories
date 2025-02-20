'use server';
import { users } from '@/app/db/schema';
import { db } from '@/app/db';
import { eq } from 'drizzle-orm';
import { auth } from "@/auth";
import bcrypt from "bcryptjs";


export async function updatePassword(email: string, oldPassword: string, newPassword: string,) {
  const session = await auth();
  if (session?.user.email !== email) {
    throw new Error('not allowed');
  }
  try {
    const existingUser = await db.query.users.findFirst({
      where: eq(users.email, email),
    });

    if (!existingUser) {
      return {
        success: false,
        message: '该用户不存在',
      };
    }
    const isMatch = await bcrypt.compare(oldPassword, existingUser.password || '');
    if (!isMatch) {
      return {
        success: false,
        message: '旧密码错误',
      };
    }

    let updateResult = null;
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(newPassword, salt);
    // 更新用户信息
    updateResult = await db.update(users)
      .set({
        password: hashedPassword,
      })
      .where(eq(users.email, email));
    return {
      success: true,
      message: '已更新',
    };

  } catch (error) {
    return {
      success: false,
      message: 'database delete error'
    }
  }
}