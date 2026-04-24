import box_lift_setup as setup
from scripts.run_planner import main


def _post_setup(setup):
    setup.contact_sampler.flip_axis_prob = 0.3


main(setup, post_setup=_post_setup)
