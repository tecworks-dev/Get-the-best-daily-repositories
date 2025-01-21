from manim import *
import numpy as np

##############################################################################
#  StarField utility
##############################################################################
class StarField(VGroup):
    """
    Creates a field of randomly placed small dots (stars) in either 2D or 3D.
    """
    def __init__(self, is_3D=False, num_stars=400, **kwargs):
        super().__init__(**kwargs)
        for _ in range(num_stars):
            x = np.random.uniform(-7, 7)
            y = np.random.uniform(-4, 4)
            z = np.random.uniform(-3, 3) if is_3D else 0
            star = Dot(point=[x, y, z], color=WHITE, radius=0.015)
            self.add(star)

##############################################################################
#  QEDJourney main scene
##############################################################################
class QEDJourney(ThreeDScene):
    def construct(self):
        # CONFIGURE THE CAMERA to have a slight tilt/angle:
        self.camera.background_color = "#000000"
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        ############################################################################
        # 1. COSMIC STARFIELD FADE-IN
        ############################################################################
        star_field = StarField(is_3D=True, num_stars=400)
        self.play(FadeIn(star_field, run_time=3))
        self.wait()

        ############################################################################
        # 2. TITLE INTRODUCTION
        ############################################################################
        main_title = Text(
            "Quantum Field Theory:\nA Journey into the Electromagnetic Interaction",
            font_size=52,
            gradient=(BLUE, YELLOW),
            weight=BOLD
        ).scale(1.0)
        main_title.set_glow_factor(0.4)  # subtle "glow" effect in newer Manim builds

        # Move title to center first
        self.play(Write(main_title), run_time=3)
        self.wait(2)

        # Then shrink and move to upper-left
        self.play(
            main_title.animate.scale(0.5).to_corner(UL),
            run_time=2
        )
        self.wait(1)

        ############################################################################
        # 3. 4D MINKOWSKI SPACETIME WIREFRAME + LIGHT CONE
        ############################################################################
        # We'll represent Minkowski space as a 3D wireframe grid plus a light cone.
        axes = ThreeDAxes(
            x_range=[-4, 4], y_range=[-4, 4], z_range=[-4, 4],
            x_length=8, y_length=8, z_length=8
        )

        ############################################################################
        # 5. ZOOM INTO ORIGIN + E/B WAVE VISUALIZATION
        ############################################################################
        # We'll move the camera to the origin for a close-up on wave fields
        self.stop_ambient_camera_rotation()
        self.move_camera(frame_center=self.camera.frame_center + UP * 2 + IN * 5)
        self.wait(2)

        # Create a wave for E and B fields in 3D
        # We use ParametricFunction for a sine wave traveling along z-axis,
        # with E field in x, B field in y, for instance.
        wave_length = 5
        # Electric field in red
        e_wave = ParametricFunction(
            lambda t: axes.c2p(
                np.sin(2 * t),  # E field amplitude along x
                0,             # (y = 0, but we'll visually offset for clarity)
                t              # wave traveling along z
            ),
            t_range=[-wave_length, wave_length],
            color=RED
        )
        # Magnetic field in blue
        b_wave = ParametricFunction(
            lambda t: axes.c2p(
                0,
                np.sin(2 * t),  # B field amplitude along y
                t
            ),
            t_range=[-wave_length, wave_length],
            color=BLUE
        )

        # Label vectors E and B
        label_E = MathTex(r"\vec{E}", color=RED).move_to(e_wave.get_end())
        label_B = MathTex(r"\vec{B}", color=BLUE).move_to(b_wave.get_end())

        # 3D Arrow for direction of propagation (z-axis)
        propagation_arrow = Arrow3D(
            start=axes.c2p(0, 0, -wave_length),
            end=axes.c2p(0, 0, wave_length),
            color=YELLOW
        ).set_stroke(width=4)
        prop_label = Tex("Propagation (z-axis)").set_color(YELLOW)
        prop_label.next_to(propagation_arrow.get_end(), UP + RIGHT)

        self.play(
            LaggedStart(
                Create(e_wave),
                Create(b_wave),
                lag_ratio=0.5,
                run_time=3
            )
        )
        self.play(
            FadeIn(label_E, shift=RIGHT),
            FadeIn(label_B, shift=RIGHT),
            Create(propagation_arrow),
            FadeIn(prop_label),
            run_time=3
        )
        self.wait(2) 