import math
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
        self.assertAlmostEqual(defaults["jammer_x"], 7.5500, delta=0.001)
        self.assertAlmostEqual(defaults["jammer_y"], -8.4500, delta=0.001)
        self.assertAlmostEqual(defaults["jammer_z"], 1.4500, delta=0.001)

    def test_rgb_and_depth_camera_resolutions_match(self):
        package_dir = os.path.dirname(os.path.dirname(__file__))
        world_file = os.path.join(package_dir, "worlds", "drone_world.sdf")

        root = ET.parse(world_file).getroot()
        for sensor_name in ("camera_sensor", "depth_camera_sensor"):
            sensor = root.find(f".//sensor[@name='{sensor_name}']")
            self.assertIsNotNone(sensor)
            self.assertEqual(sensor.findtext("camera/image/width"), "1280")
            self.assertEqual(sensor.findtext("camera/image/height"), "720")

    def test_background_building_is_behind_flight_center_from_camera(self):
        package_dir = os.path.dirname(os.path.dirname(__file__))
        world_file = os.path.join(package_dir, "worlds", "drone_world.sdf")
        root = ET.parse(world_file).getroot()

        building = root.find(".//model[@name='background_building']")
        self.assertIsNotNone(building)
        self.assertEqual(building.findtext("static"), "true")

        building_pose = [float(value) for value in building.findtext("pose").split()]
        self.assertAlmostEqual(building_pose[5], -math.pi / 4.0, delta=0.001)

        defaults = scene_defaults_from_sdf(world_file)
        camera_xy = (defaults["camera_x"], defaults["camera_y"])
        camera_to_center = (-camera_xy[0], -camera_xy[1])
        camera_to_building = (
            building_pose[0] - camera_xy[0],
            building_pose[1] - camera_xy[1],
        )
        center_distance_squared = sum(value * value for value in camera_to_center)
        depth_ratio = sum(
            camera_to_center[index] * camera_to_building[index]
            for index in range(2)
        ) / center_distance_squared
        lateral_offset = abs(
            camera_to_center[0] * camera_to_building[1]
            - camera_to_center[1] * camera_to_building[0]
        ) / math.sqrt(center_distance_squared)

        self.assertGreater(depth_ratio, 1.0)
        self.assertLess(lateral_offset, 0.05)

    def test_radio_jammer_and_hidden_camera_cone_exist(self):
        package_dir = os.path.dirname(os.path.dirname(__file__))
        world_file = os.path.join(package_dir, "worlds", "drone_world.sdf")
        root = ET.parse(world_file).getroot()

        jammer = root.find(".//model[@name='radio_jammer']")
        beam = root.find(".//model[@name='radio_jamming_beam']")
        cone = beam.find(".//visual[@name='jamming_cone']")

        self.assertEqual(jammer.findtext("static"), "true")
        self.assertEqual(beam.findtext("static"), "false")
        self.assertEqual(cone.findtext("visibility_flags"), "0x02")
        self.assertEqual(cone.findtext("geometry/cone/length"), "22.0")
        self.assertEqual(cone.findtext("geometry/cone/radius"), "4.5")
        self.assertEqual(cone.findtext("transparency"), "0.72")


if __name__ == "__main__":
    unittest.main()
