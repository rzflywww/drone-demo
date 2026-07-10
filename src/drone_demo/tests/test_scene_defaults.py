import os
import unittest

from drone_demo.scene_config import scene_defaults_from_sdf


class SceneDefaultsTest(unittest.TestCase):
    def test_laser_controller_defaults_match_world_file(self):
        package_dir = os.path.dirname(os.path.dirname(__file__))
        world_file = os.path.join(package_dir, "worlds", "drone_world.sdf")

        defaults = scene_defaults_from_sdf(world_file)

        self.assertAlmostEqual(defaults["camera_x"], 7.8925, delta=0.001)
        self.assertAlmostEqual(defaults["camera_y"], -7.8925, delta=0.001)
        self.assertAlmostEqual(defaults["camera_z"], 1.5434, delta=0.001)
        self.assertAlmostEqual(defaults["camera_pitch"], 0.044, delta=0.001)
        self.assertAlmostEqual(defaults["camera_yaw"], 2.356, delta=0.001)
        self.assertAlmostEqual(defaults["weapon_x"], 7.9601, delta=0.001)
        self.assertAlmostEqual(defaults["weapon_y"], -7.4600, delta=0.001)
        self.assertAlmostEqual(defaults["weapon_z"], 1.500, delta=0.001)


if __name__ == "__main__":
    unittest.main()
