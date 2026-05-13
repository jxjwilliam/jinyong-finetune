from __future__ import annotations

import unittest

from scripts.video.jinyong_video_pipeline import (
    PromptTranslationResult,
    extract_job_id,
    extract_video_url,
    parse_translation_result,
)


class ParseTranslationResultTest(unittest.TestCase):
    def test_parse_translation_result_with_markdown_json(self) -> None:
        raw = """```json
{
  "video_prompt": "A lone swordsman on a misty cliff at dawn.",
  "image_prompt": "wuxia hero on misty cliff, dawn",
  "recommended_model": "kling_3",
  "camera": "slow pan",
  "mood": "epic",
  "duration": 5
}
```"""
        parsed = parse_translation_result(raw)
        self.assertIsInstance(parsed, PromptTranslationResult)
        self.assertEqual(parsed.recommended_model, "kling_3")
        self.assertEqual(parsed.duration, 5)

    def test_parse_translation_result_rejects_invalid_duration(self) -> None:
        raw = """{
  "video_prompt": "A lone swordsman on a misty cliff at dawn.",
  "image_prompt": "wuxia hero on misty cliff, dawn",
  "recommended_model": "kling_3",
  "camera": "slow pan",
  "mood": "epic",
  "duration": 7
}"""
        with self.assertRaises(ValueError):
            parse_translation_result(raw)


class NanoBananaResponseParsingTest(unittest.TestCase):
    def test_extract_job_id_supports_multiple_shapes(self) -> None:
        self.assertEqual(extract_job_id({"id": "job-1"}), "job-1")
        self.assertEqual(extract_job_id({"job_id": "job-2"}), "job-2")
        self.assertEqual(extract_job_id({"data": {"id": "job-3"}}), "job-3")

    def test_extract_video_url_supports_nested_shape(self) -> None:
        body = {"status": "completed", "output": {"video": {"url": "https://example.com/x.mp4"}}}
        self.assertEqual(extract_video_url(body), "https://example.com/x.mp4")

    def test_extract_video_url_rejects_missing(self) -> None:
        with self.assertRaises(ValueError):
            extract_video_url({"status": "completed"})


if __name__ == "__main__":
    unittest.main()
