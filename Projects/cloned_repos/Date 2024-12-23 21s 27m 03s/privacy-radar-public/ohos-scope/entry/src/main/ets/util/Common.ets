// Common.ets
import { display } from '@kit.ArkUI';

const TAG: string = 'ScanKit common';
// Camera preview stream aspect ratio. The common ratio is 1, 4/3, or 16/9. You can customize the ratio.
let ratios: Array<number> = [1, 4 / 3, 16 / 9];

export interface xComDisplay {
  width: number,
  height: number,
  offsetX: number,
  offsetY: number
}

// Obtain the optimal value of ratios.
function getRatio(ratio: number): number {
  let setRatio: number = ratio < 1 ? (1 / ratio) : ratio;
  let result: number = ratios[0];
  let minNo: number = Math.abs(setRatio - ratios[0]);
  for (let i = 1; i < ratios.length; i++) {
    if (Math.abs(setRatio - ratios[i]) < minNo) {
      minNo = Math.abs(setRatio - ratios[i]);
      result = ratios[i];
    }
  }
  return result;
}

// Obtain the xComponent size in full screen mode.
export function getXComDisplay(): xComDisplay {
  // Obtain the screen size.
  let displayClass = display.getDefaultDisplaySync();
  let width: number = px2vp(displayClass.width);
  let height: number = px2vp(displayClass.height);
  let result: xComDisplay = {
    width: 360,
    height: 640,
    offsetX: 0,
    offsetY: 0
  };
  let defaultRatio: number = getRatio(height / width);
  console.info(TAG, `defaultRatio: ${defaultRatio}`);
  if (height / width >= defaultRatio) {
    result.height = height;
    result.width = height / defaultRatio;
    result.offsetX = (width - result.width) / 2;
    result.offsetY = 0;
  } else {
    result.width = width;
    result.height = width * defaultRatio;
    result.offsetX = 0;
    result.offsetY = (height - result.height) / 2;
  }
  return result;
}