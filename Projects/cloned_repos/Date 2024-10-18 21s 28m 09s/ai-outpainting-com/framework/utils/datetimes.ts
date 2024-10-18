import dayjs from 'dayjs'

export function startOfDate(date?: dayjs.Dayjs, unit: dayjs.UnitType = 'd') {
  return dayjs(date).startOf(unit).toDate()
}

export function endOfDate(date?: dayjs.Dayjs, unit: dayjs.UnitType = 'd') {
  return dayjs(date).endOf(unit).toDate()
}
