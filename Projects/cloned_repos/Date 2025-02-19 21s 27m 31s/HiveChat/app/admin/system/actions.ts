'use server';
import { db } from '@/app/db';
import { eq } from 'drizzle-orm'
import { appSettings } from '@/app/db/schema';
import { auth } from '@/auth';

export const fetchAppSettings = async (key: string) => {
  const result = await db.query.appSettings
    .findFirst({
      where: eq(appSettings.key, key)
    });
  return result?.value;
}

export const adminAndSetAppSettings = async (key: string, newValue: string): Promise<{
  status: string;
  message?: string;
}> => {
  const session = await auth();
  if (!session?.user.isAdmin) {
    return {
      status: 'fail',
      message: '401 not allowd'
    }
  }
  return setAppSettings(key, newValue);
}

export const setAppSettings = async (key: string, newValue: string): Promise<{
  status: string;
  message?: string;
}> => {

  const result = await db.query.appSettings
    .findFirst({
      where: eq(appSettings.key, key)
    });
  if (result) {
    await db.update(appSettings)
      .set({
        value: newValue,
        updatedAt: new Date(),
      })
      .where(eq(appSettings.key, 'isRegistrationOpen'));
  } else {
    await db.insert(appSettings).values({
      key: key,
      value: newValue,
      updatedAt: new Date()
    });
  }
  return {
    status: 'success'
  }
}