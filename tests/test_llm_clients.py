import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import sys
import os
import base64

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from llm.anthropic_client import AnthropicClient

class TestAnthropicClient(unittest.TestCase):
    """Test suite for the AnthropicClient."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.client = AnthropicClient(api_key="test_key")

    def tearDown(self):
        self.loop.close()

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.get', new_callable=AsyncMock)
    def test_query_with_image(self, mock_get, mock_post):
        """Test that the client correctly constructs a multi-modal request."""
        # --- Setup Mocks ---
        # Mock the image download
        mock_image_data = b"fake_image_bytes"
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_image_data
        mock_get.return_value.raise_for_status = MagicMock()

        # Mock the LLM API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "content": [{"type": "text", "text": '{"response": "ok"}'}],
            "usage": {"input_tokens": 100, "output_tokens": 20}
        }
        mock_post.return_value.raise_for_status = MagicMock()

        # --- Execute ---
        image_url = "http://example.com/test.jpg"
        prompt = "Describe this image"
        self.loop.run_until_complete(self.client.query(prompt, image_url=image_url))

        # --- Assertions ---
        mock_get.assert_called_once_with(image_url, follow_redirects=True)
        mock_post.assert_called_once()

        # Check the structure of the data sent to the LLM API
        sent_data = mock_post.call_args.kwargs['json']
        self.assertEqual(len(sent_data['messages']), 1)

        message_content = sent_data['messages'][0]['content']
        self.assertEqual(len(message_content), 2) # Should have two parts: image and text

        image_part = message_content[0]
        text_part = message_content[1]

        self.assertEqual(image_part['type'], 'image')
        self.assertEqual(image_part['source']['type'], 'base64')
        self.assertEqual(image_part['source']['media_type'], 'image/jpeg')

        expected_base64 = base64.b64encode(mock_image_data).decode('utf-8')
        self.assertEqual(image_part['source']['data'], expected_base64)

        self.assertEqual(text_part['type'], 'text')
        self.assertEqual(text_part['text'], prompt)

if __name__ == '__main__':
    unittest.main()
