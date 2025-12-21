"""
STT (Speech-to-Text) 기능 테스트

OpenAI Whisper API를 사용한 음성 변환 기능 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from streamlits.components import transcribe_audio


class TestSTTFunctionality:
    """STT 기능 테스트"""

    @patch('streamlits.components.OpenAI')
    @patch('config.settings.api_config')
    def test_transcribe_audio_success(self, mock_config, mock_openai_class):
        """정상적인 음성 변환 테스트"""
        # Mock 설정
        mock_config.openai_api_key = "test-api-key"

        # OpenAI 클라이언트 Mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Transcription 결과 Mock
        mock_transcript = MagicMock()
        mock_transcript.text = "안녕하세요 테스트입니다"
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        # 테스트 실행
        test_audio = b"fake audio bytes"

        with patch('builtins.open', create=True) as mock_open:
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('os.unlink'):
                    # 임시 파일 Mock
                    mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                    mock_temp.return_value.__enter__.return_value.write = MagicMock()

                    result = transcribe_audio(test_audio)

        # 검증
        assert result == "안녕하세요 테스트입니다"
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch('config.settings.api_config')
    def test_transcribe_audio_no_api_key(self, mock_config):
        """API Key가 없을 때 테스트"""
        mock_config.openai_api_key = None

        # 테스트 실행
        result = transcribe_audio(b"fake audio")

        # 검증
        assert result is None

    @patch('streamlits.components.OpenAI')
    @patch('config.settings.api_config')
    def test_transcribe_audio_exception(self, mock_config, mock_openai_class):
        """예외 발생 시 처리 테스트"""
        mock_config.openai_api_key = "test-api-key"
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # 예외 발생 Mock
        mock_client.audio.transcriptions.create.side_effect = Exception("API Error")

        with patch('builtins.open', create=True):
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                mock_temp.return_value.__enter__.return_value.write = MagicMock()

                result = transcribe_audio(b"fake audio")

        # 검증
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
