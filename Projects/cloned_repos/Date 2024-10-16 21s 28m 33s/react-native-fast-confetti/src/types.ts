export type FlakeSize = {
  width: number;
  height: number;
};

export type ConfettiProps = {
  /**
   * @description number of confetti pieces to render.
   * @default 200
   */
  count?: number;
  /**
   * @description the size confetti's flake.
   */
  flakeSize?: FlakeSize;
  /**
   * @description The width of the confetti's container.
   * @default SCREEN_WIDTH
   */
  width?: number;
  /**
   * @description The height of the confetti's container.
   * @default SCREEN_HEIGHT
   */
  height?: number;
  /**
   * @description The duration of confetti animation (milliseconds).
   * @default 8000
   */
  duration?: number;
  /**
   * @description Wether the animation should play on mount.
   * @default true
   */
  autoplay?: boolean;
  /**
   * @description The array of confetti flakes color.
   */
  colors?: string[];
  /**
   * @description The delay in milliseconds before the confetti animation starts automatically after initialization.
   * @default 0
   */
  autoStartDelay?: number;
  /**
   * @description Should the confetti flakes fade out as they reach the bottom.
   */
  fadeOutOnEnd?: boolean;
  /**
   * @description The approximate space between confetti flakes vertically. Lower value results in denser confetti.
   * @default 30
   */
  verticalSpacing?: number;
  /**
   * @description A callback that is called when the falling animation starts.
   */
  onAnimationStart?: () => void;
  /**
   * @description A callback that is called when the falling animation starts.
   */
  onAnimationEnd?: () => void;
};

export type ConfettiMethods = {
  /**
   * @description start the animation from the beginning
   */
  restart: () => void;
  /**
   * @description pause the animation
   */
  pause: () => void;
  /**
   * @description reset the animation and not play it
   */
  reset: () => void;
  /**
   * @description resume the animation from where it paused
   */
  resume: () => void;
};
