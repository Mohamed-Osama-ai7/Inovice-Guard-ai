import unittest


class TestUIImports(unittest.TestCase):
    def test_overview_import_without_modulenotfounderror(self):
        """Verify src.ui.views.overview can be imported cleanly without relying on app_backup or non-existent modules."""
        try:
            from src.ui.views.overview import render_overview
            from src.ui.data import build_dashboard_dataset, get_dashboard_dataset
        except ModuleNotFoundError as e:
            self.fail(f"Importing src.ui.views.overview failed with ModuleNotFoundError: {e}")

    def test_all_ui_views_imports(self):
        """Verify all view modules in src.ui.views import cleanly."""
        from src.ui.views.ai_insights import render_ai_recommendations
        from src.ui.views.customer_search import render_customer_search
        from src.ui.views.overview import render_overview
        from src.ui.views.receivables import render_collections
        from src.ui.views.revenue_intelligence import render_revenue_forecast, render_repurchase_risk
        from src.ui.views.system import render_data_quality

        self.assertTrue(callable(render_overview))
        self.assertTrue(callable(render_ai_recommendations))
        self.assertTrue(callable(render_customer_search))
        self.assertTrue(callable(render_collections))
        self.assertTrue(callable(render_revenue_forecast))
        self.assertTrue(callable(render_repurchase_risk))
        self.assertTrue(callable(render_data_quality))



if __name__ == "__main__":
    unittest.main()
