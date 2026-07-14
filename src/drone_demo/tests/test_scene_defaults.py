import os
import unittest
import xml.etree.ElementTree as ET

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
        self.assertEqual(defaults["image_width"], 1280.0)
        self.assertEqual(defaults["image_height"], 720.0)
        self.assertAlmostEqual(defaults["horizontal_fov"], 1.047, delta=0.001)
        self.assertAlmostEqual(defaults["weapon_x"], 7.9601, delta=0.001)
        self.assertAlmostEqual(defaults["weapon_y"], -7.4600, delta=0.001)
        self.assertAlmostEqual(defaults["weapon_z"], 1.500, delta=0.001)

    def test_rgb_and_depth_camera_resolutions_match(self):
        package_dir = os.path.dirname(os.path.dirname(__file__))
        world_file = os.path.join(package_dir, "worlds", "drone_world.sdf")

        root = ET.parse(world_file).getroot()
        for sensor_name in ("camera_sensor", "depth_camera_sensor"):
            sensor = root.find(f".//sensor[@name='{sensor_name}']")
            self.assertIsNotNone(sensor)
            self.assertEqual(sensor.findtext("camera/image/width"), "1280")
            self.assertEqual(sensor.findtext("camera/image/height"), "720")


if __name__ == "__main__":
    unittest.main()
