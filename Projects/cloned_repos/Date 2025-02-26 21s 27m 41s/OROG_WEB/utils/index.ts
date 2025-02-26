import Decimal from 'decimal.js';

// 数据脱敏
export const mobileHidden = (
  value: string,
  start: number = 10,
  end: number = 4,
  d: number = 3
) => {
  const n = start - 1 + d;
  if (value) {
    const valueArray = value.split("");
    for (let i = start; i < valueArray.length - end; i++) {
      valueArray[i] = ".";
      if (i > n) {
        valueArray[i] = "";
      }
    }
    return valueArray.join("");
  }
};
// 复制函数
export const handleCopyClick = (data: string | number) => {
  try {
    data = data + "";
    const textField = document.createElement("textarea");
    textField.innerText = data;
    document.body.appendChild(textField);
    textField.select();
    document.execCommand("copy");
    textField.remove();
    return true;
  } catch (e) {
    console.log(e);
    return false;
  }
};

// 转换时间差为自定义格式
export const formatDuration = (
  milliseconds: number,
  onlyLargestUnit = false
) => {
  const currentTimestamp = Math.floor(Date.now() / 1000); // 当前时间戳（秒）
  const diffInSeconds = Math.abs(currentTimestamp - milliseconds); // 计算差值秒数

  const minutes = Math.floor(diffInSeconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  const months = Math.floor(days / 30);
  const years = Math.floor(months / 12);

  const remainMonths = months % 12;
  const remainDays = days % 30;
  const remainHours = hours % 24;
  const remainMinutes = minutes % 60;
  const remainSeconds = diffInSeconds % 60;

  const parts: { unit: string; value: number }[] = [
    { unit: "y", value: years },
    { unit: "mo", value: remainMonths },
    { unit: "d", value: remainDays },
    { unit: "h", value: remainHours },
    { unit: "m", value: remainMinutes },
    { unit: "s", value: remainSeconds },
  ];

  if (onlyLargestUnit) {
    // 找到第一个非零时间单位
    const largestUnit = parts.find((part) => part.value > 0);
    return largestUnit ? `${largestUnit.value}${largestUnit.unit}` : "0s";
  } else {
    // 保留当前功能：展示所有非零单位
    const result = parts
      .filter((part) => part.value > 0)
      .map((part) => `${part.value}${part.unit}`);
    return result.length > 0 ? result.join(" ") : "0s";
  }
};
// 日期格式化 月/日/年 时:分
export function formatDate(input: Date | string | number): string {
  let date;

  // 处理不同类型的输入
  if (typeof input === "string" || typeof input === "number") {
    date = new Date(~~input * 1000);
  } else if (input instanceof Date) {
    date = input;
  } else {
    throw new Error("Invalid input type. Expected a Date, string, or number.");
  }

  // 格式化日期
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const year = date.getFullYear();
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");

  return `${month}/${day}/${year} ${hours}:${minutes}`;
}

// 数字转换为带单位的字符串 取绝对值
/**
 * 格式化大数字，使其更易于阅读。
 *
 * 此函数将大数字格式化为带有单位的字符串，适用于千（k）、百万（M）、十亿（B）级别。
 * 它可以根据数字的大小自动选择合适的单位，并对小数部分进行截断或格式化。
 *
 * @param num 要格式化的数字。
 * @param nums 小数点前导零的最小数量，默认为4。
 * @param i18nUnits 自定义单位，用于替换默认的"k"、"M"、"B"单位。
 * @returns 格式化后的数字字符串。
 */
export const formatLargeNumber = (
  num: number,
  nums: number = 4,
  precision: number = 4,
  i18nUnits?: Record<"k" | "M" | "B", string>
) => {
  if (!num) return "0";

  const absNum = new Decimal(num).abs();
  const defaultUnits: Record<"k" | "M" | "B", string> = {
    k: "k",
    M: "M",
    B: "B",
  };
  const units = { ...defaultUnits, ...i18nUnits };

  // **小于 1000，保留 4 位小数，处理 0.0000x 格式**
  if (absNum.lt(1000)) {
    const str = absNum.toFixed(10).replace(/0+$/, ""); // 去掉末尾 0
    const match = str.match(/^0\.(0*)(\d+)/);

    if (match) {
      const [, leadingZeros, significantPart] = match;
      const zeroCount = leadingZeros.length;

      if (zeroCount >= nums) {
        // 生成下标 0
        const subscriptZeros = Array.from(zeroCount.toString())
          .map((digit) => String.fromCharCode(0x2080 + parseInt(digit)))
          .join("");

        return `0.0${subscriptZeros}${significantPart
          .slice(0, precision)
          .replace(/0+$/, "")}`;
      }
    }

    // **普通情况，保留 4 位小数**
    const [integerPart, fractionalPart = ""] = absNum.toFixed(10).split(".");
    const truncatedFraction = fractionalPart
      .slice(0, precision)
      .replace(/0+$/, "");

    return truncatedFraction.length > 0
      ? `${integerPart}.${truncatedFraction}`
      : integerPart;
  }

  // **大于等于 1000，保留 1 位小数，带单位**
  let unitIndex = -1;
  let scaledNum = absNum;
  const unitKeys: ("k" | "M" | "B")[] = ["k", "M", "B"];

  while (scaledNum.gte(1000) && unitIndex < unitKeys.length - 1) {
    scaledNum = scaledNum.div(1000);
    unitIndex++;
  }

  // **四舍五入改为直接截断**
  const [integerPart, fractionalPart = ""] = scaledNum.toFixed(10).split(".");
  const truncatedFraction = fractionalPart.slice(0, 1); // **保留 1 位小数**

  const formattedScaledNum = `${integerPart}.${truncatedFraction}`.replace(
    /\.?0+$/,
    ""
  ); // 去掉末尾 0

  const unit = unitKeys[unitIndex] ? units[unitKeys[unitIndex]] : "";
  return `${formattedScaledNum}${unit}`;
};
// 取绝对值 展示价格
export const formatLeadingZeros = (num: number) => {
  const absNum = new Decimal(num).abs();

  const decimalPart = absNum.toFixed(10).split(".")[1] || "";
  const leadingZerosMatch = decimalPart.match(/^0{4,}/);

  if (leadingZerosMatch) {
    const match = absNum.toFixed(10).match(/^0\.(0*)(\d+)/);
    if (!match) return absNum.toFixed(4); // 兜底，防止解析失败

    const [, leadingZeros, significantPart] = match;
    const zeroCount = leadingZeros.length;

    const subscriptZeros = Array.from(zeroCount.toString())
      .map((digit) => String.fromCharCode(0x2080 + parseInt(digit)))
      .join("");

    const formattedSignificant = significantPart.slice(0, 4);
    return `0.0${subscriptZeros}${formattedSignificant}`;
  }

  const formattedNumber = absNum.toFixed(10);
  const [integerPart, fractionalPart] = formattedNumber.split(".");

  const trimmedFraction = fractionalPart ? fractionalPart.slice(0, 4) : "";
  const result = fractionalPart
    ? `${integerPart}.${trimmedFraction}`
    : integerPart;

  return `${result}`;
};
// 用于显示百分比 第二个参数是否取绝对值
export const multiplyAndTruncate = (
  num: number,
  isAbs: boolean = false
): string => {
  const multiplied = new Decimal(num).mul(100);
  const finalNum = isAbs ? multiplied.abs() : multiplied;
  const str = finalNum.toFixed(10); // 避免科学计数法

  const decimalIndex = str.indexOf(".");
  if (decimalIndex === -1) return str;

  const integerPart = str.substring(0, decimalIndex);
  const decimalPart = str.substring(decimalIndex + 1, decimalIndex + 3);
  const formattedDecimalPart = decimalPart.replace(/0+$/, "");

  return formattedDecimalPart.length > 0
    ? `${integerPart}.${formattedDecimalPart}`
    : integerPart;
};
/**
 * 函数addDecimal用于将两个数字或字符串类型的数字相加，返回相加结果的字符串形式。
 * 此函数特别适用于需要进行高精度计算的场景，可以避免JavaScript中浮点数计算的精度问题。
 *
 * @param num1 第一个操作数，可以是数字或字符串类型的数字。
 * @param num2 第二个操作数，可以是数字或字符串类型的数字。
 * @returns 返回相加结果的字符串形式。
 */
export const addDecimal = (num1: number | string, num2: number | string) => {
  return toNonExponential(new Decimal(num1).plus(num2).toString());
};

/**
 * 减去两个十进制数并返回结果。
 *
 * 此函数接受两个参数，可以是数字或字符串。它使用Decimal库来处理减法操作，
 * 以确保精确的十进制计算。这对于处理货币或其他需要高精度计算的场景非常有用。
 *
 * @param num1 第一个数，可以是数字或字符串。
 * @param num2 第二个数，可以是数字或字符串。
 * @returns 返回两个数相减的结果，以字符串形式表示。
 */
export const subtractDecimal = (
  num1: number | string,
  num2: number | string
) => {
  // 使用Decimal库进行减法运算，并将结果转换为字符串格式返回
  return toNonExponential(new Decimal(num1).minus(num2).toString());
};

/**
 * 多乘以两个十进制数。
 *
 * 此函数旨在提供一种精确的乘法运算，适用于包含小数的数值计算。
 * 它使用了`Decimal`库来处理运算，以避免JavaScript中浮点数计算的精度问题。
 *
 * @param num1 第一个乘数，可以是数字或字符串形式的十进制数。
 * @param num2 第二个乘数，可以是数字或字符串形式的十进制数。
 * @returns 返回两个乘数的精确乘积，以字符串形式表示。
 */
export const multiplyDecimal = (
  num1: number | string,
  num2: number | string
): string => {
  return toNonExponential(new Decimal(num1).times(num2));
};
/**
 * 分割小数点进行除法运算。
 *
 * 该函数接受两个参数，可以是数字或字符串。它使用`Decimal`库来处理除法运算，
 * 旨在提供更高精度的除法结果，避免JavaScript中浮点数计算的精度问题。
 *
 * @param num1 被除数，可以是数字或字符串。
 * @param num2 除数，可以是数字或字符串。
 * @returns 返回除法运算的结果，以字符串形式表示，确保了高精度。
 */
export const divideDecimal = (
  num1: number | string,
  num2: number | string,
  numOrStt: boolean = true
) => {
  return numOrStt
    ? toNonExponential(new Decimal(num1).dividedBy(num2))
    : new Decimal(num1).dividedBy(num2);
};
// 将科学计数法转换为字符串
export function toNonExponential(num: any) {
  var m = num.toExponential().match(/\d(?:\.(\d*))?e([+-]\d+)/);
  return num.toFixed(Math.max(0, (m[1] || "").length - m[2]));
}
