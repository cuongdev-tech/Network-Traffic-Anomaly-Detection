"""Optional native-Qt smoke check: python -m unittest giao_dien.test_gui."""
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    from giao_dien.main import Dashboard, configure_app
except ImportError as error:
    raise unittest.SkipTest("Cần cài giao_dien/requirements.txt để chạy kiểm tra Qt.") from error


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_app(cls.app)

    def test_missing_results_can_render_and_refresh_every_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            window = Dashboard(temporary)
            try:
                window.show()
                for dataset in range(window.dataset.count()):
                    window.dataset.setCurrentIndex(dataset)
                    window.refresh()
                    for page in range(window.navigation.count()):
                        window.navigation.setCurrentRow(page)
                        self.app.processEvents()
                        self.assertFalse(window.grab().isNull())
                    self.assertTrue(window.results.empty)
                    self.assertEqual(window.model_choice.count(), 0)
                    self.assertEqual(window.eda_image_choice.count(), 0)
                    self.assertTrue(window.errors.text())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
