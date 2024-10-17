import dayjs from 'dayjs'

import relativeTime from 'dayjs/plugin/relativeTime'
import duration from 'dayjs/plugin/duration'

dayjs.extend(duration)
dayjs.extend(relativeTime)

/**
 * 格式化日期
 * @param date
 */
export function formatDate(date: dayjs.Dayjs | Date | null | undefined) {
  if (!date) return ''
  return dayjs(date).format('YYYY-MM-DD')
}

/**
 * 格式化日期时间
 * @param date
 */
export function formatDateTime(date: dayjs.Dayjs | Date | null | undefined) {
  if (!date) return ''
  return dayjs(date).format('YYYY-MM-DD HH:mm:ss')
}

/**
 * 移除非法url 字符
 * @param str
 */
export function sanitizeUrl(str: string | null | undefined) {
  if (!str) return '';

  // 规范化字符串，确保统一处理斜杠和点
  const normalizedStr = str.replace(/\\/g, '/').toLowerCase();

  // 分离文件名和扩展名
  const parts = normalizedStr.split('.');
  const extension = parts.pop(); // 获取扩展名
  const fileNameWithoutExtension = parts.join('.'); // 获取无扩展名的部分

  // 仅对文件名部分进行非法字符替换，保留扩展名
  const sanitizedFileName = fileNameWithoutExtension.replace(/[^-\w\u4e00-\u9fa5]+/g, '');

  // 重新组合文件名和扩展名
  return [sanitizedFileName, extension].filter(Boolean).join('.').trim();
}

/**
 * 格式化相对时间
 * @param time
 */
export function formatRelativeTime(
  time: dayjs.Dayjs | Date | string | null | undefined,
) {
  if (!time) return '刚刚'
  const diff = dayjs().diff(time)
  const duration = dayjs.duration(diff)
  const minutes = duration.minutes()
  const hours = duration.hours()
  const days = duration.days()
  if (days > 0) {
    return `${days} 天前`
  } else if (hours > 0) {
    return `${hours} 小时前`
  } else if (minutes > 0) {
    return `${minutes} 分钟前`
  } else {
    return '刚刚'
  }
}

export function formatToMinutes(seconds: number): string {
  const duration = dayjs.duration(seconds, 'second')
  return duration.format('mm:ss')
}

export function formatKBtoMB(b: number): string {
  const kb = b / 1024
  if (kb < 1024) {
    return `${kb}KB`
  }
  const mb = kb / 1024
  return `${mb.toFixed(2)}MB`
}
