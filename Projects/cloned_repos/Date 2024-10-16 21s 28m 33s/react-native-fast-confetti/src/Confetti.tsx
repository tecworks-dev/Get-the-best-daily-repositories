import {
  useTexture,
  Group,
  Rect,
  rect,
  useRSXformBuffer,
  Canvas,
  Atlas,
} from '@shopify/react-native-skia';
import { forwardRef, useEffect, useImperativeHandle, useMemo } from 'react';
import { StyleSheet, useWindowDimensions, View } from 'react-native';
import {
  cancelAnimation,
  interpolate,
  useDerivedValue,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';
import {
  getRandomBoolean,
  getRandomValue,
  randomColor,
  randomXArray,
} from './utils';
import {
  DEFAULT_AUTOSTART_DELAY,
  DEFAULT_BOXES_COUNT,
  DEFAULT_COLORS,
  DEFAULT_DURATION,
  DEFAULT_FLAKE_SIZE,
  DEFAULT_VERTICAL_SPACING,
} from './constants';
import type { ConfettiMethods, ConfettiProps } from './types';

export const Confetti = forwardRef<ConfettiMethods, ConfettiProps>(
  (
    {
      count = DEFAULT_BOXES_COUNT,
      flakeSize = DEFAULT_FLAKE_SIZE,
      duration = DEFAULT_DURATION,
      colors = DEFAULT_COLORS,
      autoStartDelay = DEFAULT_AUTOSTART_DELAY,
      verticalSpacing = DEFAULT_VERTICAL_SPACING,
      onAnimationEnd,
      onAnimationStart,
      width: _width,
      height: _height,
      autoplay = true,
      fadeOutOnEnd = false,
    },
    ref
  ) => {
    const progress = useSharedValue(0);
    const opacity = useDerivedValue(() => {
      if (!fadeOutOnEnd) return 1;
      return interpolate(progress.value, [0, 0.9, 1], [1, 0, 0]);
    }, [fadeOutOnEnd]);
    const running = useSharedValue(false);
    const { width: DEFAULT_SCREEN_WIDTH, height: DEFAULT_SCREEN_HEIGHT } =
      useWindowDimensions();
    const containerWidth = _width || DEFAULT_SCREEN_WIDTH;
    const containerHeight = _height || DEFAULT_SCREEN_HEIGHT;
    const columnsNum = Math.floor(containerWidth / flakeSize.width);
    const rowsNum = Math.ceil(count / columnsNum);
    const rowHeight = flakeSize.height + verticalSpacing;
    const verticalOffset = -rowsNum * rowHeight;
    const textureSize = {
      width: flakeSize.width * columnsNum,
      height: flakeSize.height * rowsNum,
    };

    const reset = () => {
      running.value = false;
      cancelAnimation(progress);
      progress.value = 0;
    };

    const restart = () => {
      progress.value = 0;
      running.value = true;
      if (autoplay)
        progress.value = withRepeat(withTiming(1, { duration }), -1, false);
      else progress.value = withTiming(1, { duration });
    };

    const resume = () => {
      if (running.value) return;

      const remaining = duration * (1 - progress.value);
      running.value = true;
      progress.value = withTiming(1, { duration: remaining }, (finished) => {
        if (finished) {
          onAnimationEnd?.();
          progress.value = 0;
          if (autoplay) {
            onAnimationStart?.();
            progress.value = withRepeat(withTiming(1, { duration }), -1, false);
          }
        }
      });
    };

    const pause = () => {
      running.value = false;
      cancelAnimation(progress);
    };

    useImperativeHandle(ref, () => ({
      pause,
      reset,
      resume,
      restart,
    }));

    const boxes = useMemo(
      () =>
        new Array(count).fill(0).map(() => ({
          clockwise: getRandomBoolean(),
          maxRotation: {
            x: getRandomValue(2 * Math.PI, 20 * Math.PI),
            z: getRandomValue(2 * Math.PI, 20 * Math.PI),
          },
          color: randomColor(colors),
          randomXs: randomXArray(5, -50, 50), // Array of randomX values for horizontal movement
          randomSpeed: getRandomValue(0.9, 1.3), // Random speed multiplier
          randomOffsetX: getRandomValue(-10, 10), // Random X offset for initial position
          randomOffsetY: getRandomValue(-10, 10), // Random Y offset for initial position
        })),
      [count, colors]
    );

    const getPosition = (index: number) => {
      'worklet';
      const x = (index % columnsNum) * flakeSize.width;
      const y = Math.floor(index / columnsNum) * rowHeight;

      return { x, y };
    };

    useEffect(() => {
      if (autoplay && !running.value) setTimeout(restart, autoStartDelay);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [autoplay]);

    const texture = useTexture(
      <Group>
        {boxes.map((box, index) => {
          const { x, y } = getPosition(index);

          return (
            <Rect
              key={box.maxRotation.x * box.maxRotation.z}
              rect={rect(x, y, flakeSize.width, flakeSize.height)}
              color={box.color}
            />
          );
        })}
      </Group>,
      textureSize
    );

    const sprites = boxes.map((_, index) => {
      const { x, y } = getPosition(index);
      return rect(x, y, flakeSize.width, flakeSize.height);
    });

    const transforms = useRSXformBuffer(count, (val, i) => {
      'worklet';
      const piece = boxes[i];
      if (!piece) return;

      const { x, y } = getPosition(i); // Already includes random offsets
      const tx = x + piece.randomOffsetX;
      const maxYMovement = -verticalOffset + containerHeight * 1.5; // Add extra to compensate for different speeds
      let ty = y + piece.randomOffsetY + verticalOffset;

      // Apply random speed to the fall height
      const fallHeight = interpolate(
        progress.value,
        [0, 1],
        [0, maxYMovement * piece.randomSpeed] // Use random speed here
      );

      // Interpolate between randomX values for smooth left-right movement
      const randomX = interpolate(
        progress.value,
        [0, 0.25, 0.5, 0.75, 1],
        piece.randomXs // Use the randomX array for horizontal movement
      );

      const rotationDirection = piece.clockwise ? 1 : -1;
      const rz = interpolate(
        progress.value,
        [0, 1],
        [0, rotationDirection * piece.maxRotation.z]
      );
      const rx = interpolate(
        progress.value,
        [0, 1],
        [0, rotationDirection * piece.maxRotation.x]
      );
      ty += fallHeight;

      const scale = Math.abs(Math.cos(rx)); // Scale goes from 1 -> 0 -> 1

      const px = flakeSize.width / 2;
      const py = flakeSize.height / 2;

      // Apply the transformation, including the flipping effect and randomX oscillation
      const s = Math.sin(rz) * scale;
      const c = Math.cos(rz) * scale;

      // Use the interpolated randomX for horizontal oscillation
      val.set(c, s, tx + randomX - c * px + s * py, ty - s * px - c * py);
    });

    return (
      <View pointerEvents="none" style={styles.container}>
        <Canvas style={styles.canvasContainer}>
          <Atlas
            image={texture}
            sprites={sprites}
            transforms={transforms}
            opacity={opacity}
          />
        </Canvas>
      </View>
    );
  }
);

const styles = StyleSheet.create({
  container: {
    height: '100%',
    width: '100%',
    position: 'absolute',
    zIndex: 1,
  },
  canvasContainer: {
    width: '100%',
    height: '100%',
  },
});
