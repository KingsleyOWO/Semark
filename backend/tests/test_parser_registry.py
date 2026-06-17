import unittest

from app.adapters.parser_registry import get_parser_specs, probe_all_parsers


class ParserRegistryTest(unittest.TestCase):
    def test_registry_contains_planned_candidates(self):
        ids = {spec.parser_id for spec in get_parser_specs()}
        self.assertIn("mineru3_pipeline", ids)
        self.assertIn("docling_standard", ids)
        self.assertIn("paddleocr_structure_v3", ids)
        self.assertIn("olmocr", ids)
        self.assertIn("marker_reference", ids)

    def test_marker_is_reference_only(self):
        marker = next(spec for spec in get_parser_specs() if spec.parser_id == "marker_reference")
        self.assertTrue(marker.reference_only)
        self.assertFalse(marker.open_source_default)

    def test_probe_shape_is_json_serializable(self):
        probes = [probe.to_dict() for probe in probe_all_parsers()]
        self.assertTrue(probes)
        self.assertIn("available", probes[0])
        self.assertIn("commands_found", probes[0])


if __name__ == "__main__":
    unittest.main()

