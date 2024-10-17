export const getRandomBoolean = () => {
  'worklet';
  return Math.random() >= 0.5;
};

export const getRandomValue = (min: number, max: number): number => {
  'worklet';
  return Math.random() * (max - min) + min;
};

export const randomColor = (colors: string[]): string => {
  'worklet';
  return colors[Math.floor(Math.random() * colors.length)] as string;
};

export const randomXArray = (num: number, min: number, max: number) => {
  'worklet';
  return new Array(num).fill(0).map(() => getRandomValue(min, max));
};
