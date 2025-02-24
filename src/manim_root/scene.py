import manim as mn

# specify quality with -ql, -qm, -qh, -qp, -qk
# play after rendering with -p
# open folder after rendering with -f
# render all scenes in file with -a
# render the last image with -s
# render a gif with --format gif
# render sections separately with --save_sections

class CreateCircle(mn.Scene):
    def construct(self):
        circle = mn.Circle()  # create a circle
        circle.set_fill(mn.PINK, opacity=0.5)  # set color and transparency

        square = mn.Square()  # create a square
        square.flip(mn.RIGHT)  # flip horizontally
        square.rotate(-3 * mn.TAU / 8)  # rotate a certain amount

        self.play(mn.Create(square))  # animate the creation of the square
        self.next_section()
        self.play(mn.Transform(square, circle))  # interpolate the square into the circle
        self.play(mn.FadeOut(square))  # fade out animation

class SquareToCircle(mn.Scene):
    def construct(self):
        circle = mn.Circle()  # create a circle
        circle.set_fill(mn.PINK, opacity=0.5)  # set color and transparency

        square = mn.Square()  # create a square
        square.rotate(mn.PI / 4)  # rotate a certain amount

        self.play(mn.Create(square))  # animate the creation of the square
        self.play(mn.Transform(square, circle))  # interpolate the square into the circle
        self.play(mn.FadeOut(square))  # fade out animation

class SquareAndCircle(mn.Scene):
    def construct(self):
        circle = mn.Circle()  # create a circle
        circle.set_fill(mn.PINK, opacity=0.5)  # set the color and transparency

        square = mn.Square()  # create a square
        square.set_fill(mn.BLUE, opacity=0.5)  # set the color and transparency

        square.next_to(circle, mn.UP, buff=0.5)  # set the position
        self.play(mn.Create(circle), mn.Create(square))  # show the shapes on screen

class AnimatedSquareToCircle(mn.Scene):
    def construct(self):
        circle = mn.Circle()  # create a circle
        square = mn.Square()  # create a square

        self.play(mn.Create(square))  # show the square on screen
        self.play(square.animate.rotate(mn.PI / 4))  # rotate the square
        self.play(mn.Transform(square, circle))  # transform the square into a circle
        self.play(
            square.animate.set_fill(mn.PINK, opacity=0.5)
        )  # color the circle on screen

class DifferentRotations(mn.Scene):
    def construct(self):
        left_square = mn.Square(color=mn.BLUE, fill_opacity=0.7).shift(2 * mn.LEFT)
        right_square = mn.Square(color=mn.GREEN, fill_opacity=0.7).shift(2 * mn.RIGHT)
        self.play(
            left_square.animate.rotate(mn.PI), mn.Rotate(right_square, angle=mn.PI), run_time=2
        )
        self.wait()