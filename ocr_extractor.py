import os
import json
from dotenv import load_dotenv
from mindee import ClientV2, InferenceParameters, InferenceResponse, PathInput

load_dotenv()

class MindeeFullParser:
    def __init__(self):
        self.api_key = os.getenv("MINDEE_API_KEY")
        if not self.api_key:
            raise ValueError("API key missing")

        self.model_id = "a3536038-e793-4f2f-a2a0-4cac5e236927"

        self.client = ClientV2(self.api_key)

    def get_full_json(self, file_path):
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        try:
            params = InferenceParameters(
                model_id=self.model_id,
                rag=True,
                raw_text=True,
                polygon=True,
                confidence=True,
            )

            input_source = PathInput(file_path)

            response = self.client.enqueue_and_get_result(
                InferenceResponse,
                input_source,
                params,
            )

            # Convert full response to JSON
            full_json = response.__dict__

            return full_json

        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    parser = MindeeFullParser()

    output = parser.get_full_json("receipt.jpg")

    with open("mindee_output.json", "w") as f:
        json.dump(output, f, indent=4, default=str)

    print("✅ Success! JSON saved.")