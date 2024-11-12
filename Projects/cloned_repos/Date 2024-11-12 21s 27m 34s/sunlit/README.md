

https://github.com/user-attachments/assets/b4b2a21b-cbfa-4d09-8475-0946cb96cb62

---

inspired by https://daylightcomputer.com/ and https://www.sunlit.place/

## components
### base 

- this is all just divs with background
- the leaves i sourced from adobe stock photos and downscaled to ~300x500 because i knew i would be blurring it anyways

```html
<div id="blinds">
<div class="shutters">
  <div class="shutter"></div>
  ...
  <div class="shutter"></div>
</div>
<div class="vertical">
  <div class="bar"></div>
  <div class="bar"></div>
</div>
</div>
```

![Screenshot 2024-11-11 at 1 51 49 PM](https://github.com/user-attachments/assets/cd9043b8-cd6f-455e-9174-fffd3ba4b416)

### progressive blur

- gotta make the diffusion effect convincing
- things closer to the wall should be less blurry than things away from the wall
- this utilizes multiple blur layers each with a mask image to constraint its effect zone

```css
#progressive-blur {
  position: absolute;
  height: 100%;
  width: 100%;
}

#progressive-blur>div {
  position: absolute;
  height: 100%;
  width: 100%;
  inset: 0;
  backdrop-filter: blur(var(--blur-amount));
  mask-image: linear-gradient(252deg, transparent, transparent var(--stop1), black var(--stop2), black);
}

#progressive-blur>div:nth-child(1) {
  --blur-amount: 6px;
  --stop1: 0%;
  --stop2: 0%;
}

#progressive-blur>div:nth-child(2) {
  --blur-amount: 12px;
  --stop1: 40%;
  --stop2: 80%;
}

#progressive-blur>div:nth-child(3) {
  --blur-amount: 48px;
  --stop1: 40%;
  --stop2: 70%;
}

#progressive-blur>div:nth-child(4) {
  --blur-amount: 96px;
  --stop1: 70%;
  --stop2: 80%;
}
```

![Screenshot 2024-11-11 at 1 52 01 PM](https://github.com/user-attachments/assets/834db461-2ae5-4528-ad74-c147c3472090)

### leaf billowing

- created a leaf billowing effect using small amounts of rotate and scale
- to add a bit more dynamism, i used svg filters and the lesser known `feTurbulence` and `feDisplacementMap` tags to add a higher octave of noise to billow individual leaves

```html
  <div id="leaves">
    <svg style="width: 0; height: 0; position: absolute;">
      <defs>
        <filter id="wind" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="fractalNoise" numOctaves="2" seed="1">
            <animate attributeName="baseFrequency" dur="16s" keyTimes="0;0.33;0.66;1"
              values="0.005 0.003;0.01 0.009;0.008 0.004;0.005 0.003" repeatCount="indefinite" />
          </feTurbulence>
          <feDisplacementMap in="SourceGraphic">
            <animate attributeName="scale" dur="20s" keyTimes="0;0.25;0.5;0.75;1" values="45;55;75;55;45"
              repeatCount="indefinite" />
          </feDisplacementMap>
        </filter>
      </defs>
    </svg>
  </div>
```

```css
#leaves {
  position: absolute;
  background-size: cover;
  background-repeat: no-repeat;
  bottom: -20px;
  right: -700px;
  width: 1600px;
  height: 1400px;
  background-image: url("./leaves.png");
  filter: url(#wind);
  animation: billow 8s ease-in-out infinite;
}

@keyframes billow {
  0% {
    transform: perspective(400px) rotateX(0deg) rotateY(0deg) scale(1);
  }

  25% {
    transform: perspective(400px) rotateX(1deg) rotateY(2deg) scale(1.02);
  }

  50% {
    transform: perspective(400px) rotateX(-4deg) rotateY(-2deg) scale(0.97);
  }

  75% {
    transform: perspective(400px) rotateX(1deg) rotateY(-1deg) scale(1.04);
  }

  100% {
    transform: perspective(400px) rotateX(0deg) rotateY(0deg) scale(1);
  }
}
```

https://github.com/user-attachments/assets/854f4321-f40e-465f-9b26-4fe663d628fa


### sunrise/sunset transitions

- the light/dark transition was a bit too abrupt so i decided to add a transition between the two
- a linear interpolation of colors wasn't super exciting so I decided to create a sunrise/sunset color animation
- these colors are just handpicked from many hours of just looking at sunsets and some experimentation, nothing too fancy here
- theres also a subtle 'glow' that is a very crude approximation of light bouncing off the floor


### 3d transform

- finally, skew + rotate + transform didnt get what i wanted because it only gives you affine transforms and i want that nice perspective distortion
- decided to derive a transform matrix for memes, this involved pulling desmos to draw some shapes
- then, we can solve for a 4d perspective transform matrix that `matrix3d` can use in css

<img width="1712" alt="Screenshot 2024-11-11 at 11 47 53 AM" src="https://github.com/user-attachments/assets/040dde51-f114-47cd-8870-f43eb4c4413e">

`matrix3d` values are calculated using the following python script:

```python
import numpy as np

def compute_homography(points_src, points_dst):
    if points_src.shape != (4, 2) or points_dst.shape != (4, 2):
        raise ValueError("Input arrays must be of shape (4,2)")

    A = np.zeros((8, 8))
    b = np.zeros(8)

    for i in range(4):
        x, y = points_src[i]
        x_prime, y_prime = points_dst[i]
        A[i * 2] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime]
        b[i * 2] = x_prime
        A[i * 2 + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime]
        b[i * 2 + 1] = y_prime

    h = np.linalg.solve(A, b)
    H = np.array([
        [h[0], h[1], 0, h[2]],
        [h[3], h[4], 0, h[5]],
        [0.0, 0.0, 1.0, 0.0],
        [h[6], h[7], 0.0, 1.0]
    ])

    return H


if __name__ == "__main__":
    src = np.array([[-1, 0], [0, 0], [0, -1], [-1, -1]]) * 500
    # day
    dst = np.array([[-1, -0.1], [0, 0], [0, -1], [-1, -1.7]]) * 500
    # night
    # dst = np.array([[-1, 0.1], [0, 0], [0, -1], [-1, -1.1]]) * 500

    # Flip y-axis in src and dst
    src[:, 1] = -src[:, 1]
    dst[:, 1] = -dst[:, 1]

    H = compute_homography(src, dst)
    print("Homography matrix:")
    print(H)

    print("Homography matrix as matrix3d in column-major order:")
    print("transform: matrix3d(")
    for col in range(4):
        values = ", ".join(f"{H[row, col]:.4f}" for row in range(4))
        print(f"    {values}," if col < 3 else f"    {values}")
    print(");")
```
