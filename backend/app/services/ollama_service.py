import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaService:
    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        try:
            payload = self._get_json("/api/tags")
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            return False

        models = payload.get("models")
        return isinstance(models, list) and len(models) > 0

    def resolve_model(self) -> str | None:
        try:
            payload = self._get_json("/api/tags")
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            return None

        models = payload.get("models")
        if not isinstance(models, list) or not models:
            return None

        available_names = [
            item.get("name")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        if self.model in available_names:
            return self.model
        return available_names[0] if available_names else None

    def generate_report(
        self,
        *,
        modality: str,
        predicted_label: str,
        confidence: float,
        probabilities: list[dict[str, float | str]],
        template_report: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        prompt = self._build_prompt(
            modality=modality,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            template_report=template_report,
        )
        active_model = self.resolve_model()
        if active_model is None:
            raise ValueError("No installed Ollama model is currently available.")
        response_payload = self._post_json(
            "/api/generate",
            {
                "model": active_model,
                "stream": False,
                "format": "json",
                "prompt": prompt,
            },
        )
        raw_response = response_payload.get("response")
        if not isinstance(raw_response, str):
            raise ValueError("Ollama response did not contain a string payload.")

        parsed = json.loads(raw_response)
        sections = parsed.get("sections")
        if not isinstance(sections, list):
            raise ValueError("Ollama JSON payload did not contain a sections list.")

        normalized_sections: list[dict[str, str]] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = section.get("title")
            body = section.get("body")
            if not isinstance(title, str) or not isinstance(body, str):
                continue
            clean_title = title.strip()
            clean_body = body.strip()
            if clean_title and clean_body:
                normalized_sections.append({"title": clean_title, "body": clean_body})

        if not normalized_sections:
            raise ValueError("Ollama returned no usable sections.")

        return normalized_sections

    def chat(
        self,
        messages: list[dict[str, str]],
        context: dict | None = None,
    ) -> str:
        """Multi-turn chat with optional scan context injected into system prompt."""
        active_model = self.resolve_model()
        if active_model is None:
            raise ValueError("No installed Ollama model is currently available.")

        system_content = (
            "You are NeuroVision AI Assistant, a knowledgeable medical imaging assistant "
            "specializing in brain tumor analysis. You help clinicians and researchers understand "
            "AI-generated brain tumor predictions from MRI and CT scans. Always use cautious, "
            "professional language. Never claim definitive diagnosis. Remind users that AI outputs "
            "require clinical validation and expert review. Be helpful, informative, and empathetic "
            "when discussing prognosis and treatment options. Keep responses concise and focused."
        )

        if context:
            modality = str(context.get("modality", "unknown")).upper()
            prediction = str(context.get("prediction", "unknown")).replace("_", " ")
            confidence = float(context.get("confidence", 0))
            location = context.get("tumor_location")
            patient = context.get("patient", "")

            context_lines = [
                "\n\nCurrent scan context (use this to inform your responses):",
                f"- Imaging modality: {modality}",
                f"- AI predicted class: {prediction}",
                f"- Model confidence: {confidence:.1%}",
            ]
            if patient:
                context_lines.append(f"- Patient identifier: {patient}")
            if location and isinstance(location, dict):
                context_lines.append(f"- Predicted activation location: {location.get('description', 'unknown')}")
                context_lines.append(f"- Quadrant: {location.get('quadrant', 'unknown')}")

            system_content += "\n".join(context_lines)

        ollama_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            *messages,
        ]

        response_payload = self._post_json(
            "/api/chat",
            {
                "model": active_model,
                "messages": ollama_messages,
                "stream": False,
            },
        )

        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama chat response missing message object.")

        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama chat response missing content string.")

        return content.strip()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from Ollama.")
        return parsed

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from Ollama.")
        return parsed

    def _build_prompt(
        self,
        *,
        modality: str,
        predicted_label: str,
        confidence: float,
        probabilities: list[dict[str, float | str]],
        template_report: list[dict[str, str]],
    ) -> str:
        probability_lines = "\n".join(
            f"- {entry['label']}: {float(entry['probability']):.4f}" for entry in probabilities
        )
        template_lines = "\n".join(
            f"{section['title']}: {section['body']}" for section in template_report
        )
        return f"""
You are generating a structured radiology-style support note for a brain imaging AI prototype.

Constraints:
- Use cautious, professional medical language.
- Do not invent patient age, sex, symptoms, lesion size, lesion side, sequence names, measurements, or pathology results.
- Do not claim definitive diagnosis or clinical validation.
- Keep the report useful for decision support only.
- Mention that complete imaging review and clinician correlation are still required.
- Return only valid JSON with this exact schema:
  {{
    "sections": [
      {{"title": "Study summary", "body": "..."}},
      {{"title": "Technique", "body": "..."}},
      {{"title": "Findings", "body": "..."}},
      {{"title": "Saliency interpretation", "body": "..."}},
      {{"title": "Differential consideration", "body": "..."}},
      {{"title": "Impression", "body": "..."}},
      {{"title": "Recommendations and limitations", "body": "..."}}
    ]
  }}

Model context:
- Modality: {modality.upper()}
- Predicted class: {predicted_label}
- Confidence: {confidence:.4f}

Class probabilities:
{probability_lines}

Template baseline report:
{template_lines}
""".strip()
