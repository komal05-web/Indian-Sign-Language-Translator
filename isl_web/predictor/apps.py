from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    def ready(self):
        """Load TF models once when Django starts — not per-request."""
        from django.conf import settings
        from .predictor_engine import ISLEngine
        ISLEngine.load_models(
            word_model_path = settings.ISL_WORD_MODEL_PATH,
            label_map_path  = settings.ISL_LABEL_MAP_PATH,
            sent_model_path = settings.ISL_SENTENCE_MODEL_PATH,
            sent_label_path = settings.ISL_SENTENCE_LABEL_PATH,
        )
