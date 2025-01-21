import re
from manim import *  # This imports all Manim objects including Text, Circle, etc.

class MLAnimationGenerator:
    def __init__(self):
        self.templates = {
            'network': self._create_network_template,
            'training': self._create_training_template,
            'comparison': self._create_comparison_template,
            'optimal_transport': self._create_optimal_transport_template,
            'particle_system': self._create_particle_system_template,
            'vector_field': self._create_vector_field_template
        }
        
    def _create_network_template(self, params):
        """Creates a neural network visualization with customizable layers"""
        class NetworkScene(Scene):
            def construct(self):
                # Parse parameters
                layer_sizes = params.get('layer_sizes', [3, 4, 3])
                title_text = params.get('title', 'Neural Network')
                
                # Create title
                title = Text(title_text).scale(0.8)
                title.to_edge(UP)
                
                # Create layers
                layers = []
                for size in layer_sizes:
                    layer = VGroup(*[Circle(radius=0.2, color=BLUE) for _ in range(size)])
                    layer.arrange(DOWN, buff=0.3)
                    layers.append(layer)
                
                # Position layers
                network = VGroup(*layers).arrange(RIGHT, buff=2)
                network.move_to(ORIGIN)
                
                # Create connections
                connections = VGroup()
                for i in range(len(layers)-1):
                    curr_layer = layers[i]
                    next_layer = layers[i+1]
                    for n1 in curr_layer:
                        for n2 in next_layer:
                            line = Line(n1.get_right(), n2.get_left(), stroke_width=1)
                            connections.add(line)
                
                # Animations
                self.play(Write(title))
                self.play(
                    LaggedStartMap(FadeIn, network, lag_ratio=0.2),
                    run_time=2
                )
                self.play(Create(connections, lag_ratio=0.1))
                self.wait()
                
        return NetworkScene

    def _create_training_template(self, params):
        """Creates a training progress visualization"""
        class TrainingScene(Scene):
            def construct(self):
                # Create progress elements
                epochs = params.get('epochs', 5)
                title = Text(params.get('title', 'Training Progress')).scale(0.8)
                title.to_edge(UP)
                
                # Create progress bar
                bar = Rectangle(height=0.4, width=6, color=BLUE)
                bar.move_to(ORIGIN)
                bar.set_fill(BLUE, opacity=0.1)
                
                # Create accuracy text
                accuracy = DecimalNumber(
                    0,
                    num_decimal_places=1,
                    include_sign=False,
                )
                accuracy_text = Text("Accuracy: ").scale(0.8)
                accuracy_label = VGroup(accuracy_text, accuracy)
                accuracy_label.arrange(RIGHT)
                accuracy_label.next_to(bar, UP)
                
                # Animations
                self.play(Write(title))
                self.play(Create(bar), Write(accuracy_label))
                
                # Training progress
                for i in range(epochs):
                    self.play(
                        bar.animate.set_fill(opacity=(i+1)/epochs),
                        accuracy.animate.set_value((i+1)*100/epochs),
                        run_time=0.5
                    )
                
                self.wait()
                
        return TrainingScene

    def _create_comparison_template(self, params):
        """Creates a side-by-side comparison visualization"""
        class ComparisonScene(Scene):
            def construct(self):
                # Create titles
                title1 = Text(params.get('title1', 'Model A')).scale(0.6)
                title2 = Text(params.get('title2', 'Model B')).scale(0.6)
                
                # Create comparison elements (e.g., accuracy bars)
                bar1 = Rectangle(height=params.get('value1', 0.6)*3, width=1, color=BLUE)
                bar2 = Rectangle(height=params.get('value2', 0.8)*3, width=1, color=RED)
                
                # Position elements
                title1.move_to(UP*2 + LEFT*2)
                title2.move_to(UP*2 + RIGHT*2)
                bar1.next_to(title1, DOWN)
                bar2.next_to(title2, DOWN)
                
                # Animations
                self.play(Write(title1), Write(title2))
                self.play(GrowFromEdge(bar1, DOWN), GrowFromEdge(bar2, DOWN))
                self.wait()
                
        return ComparisonScene

    def _create_particle_system_template(self, params):
        """Creates a particle system visualization with interpolation"""
        class ParticleScene(Scene):
            def construct(self):
                # Create initial and final distributions
                alpha_0 = VGroup(*[Dot(radius=0.05, color=BLUE) for _ in range(50)])
                alpha_0.arrange_in_grid(rows=5, cols=10, buff=0.2)
                alpha_0.shift(LEFT * 3)
                
                alpha_1 = VGroup(*[Dot(radius=0.05, color=GOLD) for _ in range(50)])
                alpha_1.arrange_in_grid(rows=5, cols=10, buff=0.2)
                alpha_1.shift(RIGHT * 3)
                
                # Create interpolation path
                def get_alpha_t(t):
                    return VGroup(*[
                        Dot(
                            radius=0.05,
                            color=interpolate_color(BLUE, GOLD, t),
                            point=interpolate(
                                alpha_0[i].get_center(),
                                alpha_1[i].get_center(),
                                t
                            )
                        )
                        for i in range(len(alpha_0))
                    ])
                
                # Animations
                self.play(Create(alpha_0), Create(alpha_1))
                self.play(
                    UpdateFromAlphaFunc(
                        alpha_0,
                        lambda m, t: m.become(get_alpha_t(t))
                    ),
                    run_time=3
                )
                self.wait()
        
        return ParticleScene

    def _create_vector_field_template(self, params):
        """Creates a vector field visualization for transport"""
        class VectorFieldScene(Scene):
            def construct(self):
                # Create vector field
                vector_field = VectorField(
                    lambda p: np.array([p[0], p[1], 0]),
                    x_range=[-4, 4, 0.5],
                    y_range=[-3, 3, 0.5],
                    color=LIGHT_GREY
                )
                
                # Create streamlines
                stream_lines = StreamLines(
                    lambda p: np.array([p[0], p[1], 0]),
                    stroke_width=2,
                    color=SILVER
                )
                
                # Animations
                self.play(Create(vector_field))
                self.play(stream_lines.create())
                self.wait()
        
        return VectorFieldScene

    def _create_optimal_transport_template(self, params):
        """Creates visualization for optimal transport map"""
        class OptimalTransportScene(Scene):
            def construct(self):
                # Create grid for transformation visualization
                grid = NumberPlane(
                    x_range=[-4, 4],
                    y_range=[-3, 3],
                    background_line_style={
                        "stroke_color": BLUE_E,
                        "stroke_width": 1,
                        "stroke_opacity": 0.3
                    }
                )
                
                # Create transformation
                def transport_function(point):
                    x, y, z = point
                    return np.array([
                        x * np.cos(y),
                        y * np.sin(x),
                        0
                    ])
                
                # Animate grid transformation
                self.play(Create(grid))
                self.play(
                    grid.animate.apply_function(transport_function),
                    run_time=3
                )
                self.wait()
        
        return OptimalTransportScene

    def parse_description(self, text):
        """Parse natural language description into scene parameters"""
        params = {}
        
        # Detect mathematical content
        if any(term in text.lower() for term in ['α', 'ν', 'gradient', 'vector field', 'particle']):
            if 'particle' in text.lower() or 'α' in text:
                params['template'] = 'particle_system'
            elif 'vector field' in text.lower() or 'ν' in text:
                params['template'] = 'vector_field'
            elif 'transform' in text.lower() or 'map' in text.lower():
                params['template'] = 'optimal_transport'
        
        return params

    def generate_scene(self, description):
        """Generate a manim scene based on the description"""
        params = self.parse_description(description)
        template_name = params.pop('template', 'particle_system')  # Default to particle system
        
        if template_name in self.templates:
            return self.templates[template_name](params)
        else:
            # Fallback to particle system if no specific template matches
            return self.templates['particle_system'](params)

# The simple create_animation function remains as a fallback
def create_animation(description):
    """Creates a basic Manim scene class"""
    class GeneratedScene(Scene):
        def construct(self):
            # Create a simple visualization
            title = Text("Optimal Transport Visualization").scale(0.8)
            title.to_edge(UP)
            
            # Create some basic shapes
            shapes = VGroup(
                Circle(radius=1, color=BLUE),
                Square(side_length=2, color=GOLD)
            ).arrange(RIGHT, buff=2)
            
            # Basic animation
            self.play(Write(title))
            self.play(Create(shapes))
            self.play(
                shapes[0].animate.morph_to_target(shapes[1]),
                run_time=2
            )
            self.wait()
            
    return GeneratedScene

# Example usage:
if __name__ == "__main__":
    # Example descriptions:
    descriptions = [
        "Show me a neural network with 4 layers",
        "Visualize training progress over 10 epochs",
        "Compare model A (75% accuracy) versus model B (85% accuracy)"
    ]
    
    # Generate scene for the first description
    Scene = create_animation(descriptions[0])