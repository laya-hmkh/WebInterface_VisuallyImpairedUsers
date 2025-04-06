import torch
import soundfile as sf
import os
import logging
import re
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self.model = None
        self.symbols = None
        self.sample_rate = None
        self.apply_tts = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the TTS model with error handling"""
        try:
            logger.info("Loading Silero TTS model...")
            (self.model, self.symbols, 
             self.sample_rate, _, 
             self.apply_tts) = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='lj_16khz', 
            )
            self.model = self.model.to("cpu")
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Could not initialize TTS engine")

    def _preprocess_text(self, text: str) -> str:
        """Convert numbers and special characters to spoken form"""
        # Convert numbers to words
        text = re.sub(r'(\d+)', lambda x: self._number_to_words(x.group()), text)
        
        # Handle common symbols
        symbol_map = {
            '%': 'percent',
            '$': 'dollars',
            '#': 'number',
            '&': 'and',
            '@': 'at'
        }
        for sym, word in symbol_map.items():
            text = text.replace(sym, f' {word} ')
        
        # Clean up whitespace
        return ' '.join(text.split())

    def _number_to_words(self, num_str: str) -> str:
        """Convert numbers to their spoken equivalent"""
        try:
            num = int(num_str)
            if 0 <= num < 20:
                return [
                    'zero', 'one', 'two', 'three', 'four', 
                    'five', 'six', 'seven', 'eight', 'nine',
                    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'
                ][num]
            elif 20 <= num < 100:
                tens = [
                    'twenty', 'thirty', 'forty', 'fifty', 
                    'sixty', 'seventy', 'eighty', 'ninety'
                ][(num // 10) - 2]
                return tens + (' ' + self._number_to_words(num % 10) if num % 10 != 0 else '')
            elif 100 <= num < 1000:
                return self._number_to_words(num // 100) + ' hundred' + (
                    ' ' + self._number_to_words(num % 100) if num % 100 != 0 else '')
            else:
                return num_str  # Fallback for large numbers
        except:
            return num_str  # Return original if conversion fails

    def text_to_speech(self, text: str, output_path: str = "static/output.wav") -> Optional[str]:
        """
        Convert text to speech with enhanced number/symbol handling
        
        Args:
            text: Input text to convert (can contain numbers/symbols)
            output_path: Where to save the WAV file
            
        Returns:
            Path to audio file or None if failed
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid text input")
            raise ValueError("Text must be a non-empty string")

        try:
            # Preprocess text (convert numbers, symbols)
            processed_text = self._preprocess_text(text)
            logger.info(f"Processed text for TTS: {processed_text}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Generate and save audio
            audio = self.apply_tts(
                texts=[processed_text],
                model=self.model,
                sample_rate=self.sample_rate,
                symbols=self.symbols,
                device="cpu"
            )[0].numpy()

            sf.write(output_path, audio, self.sample_rate)
            logger.info(f"Audio saved to {output_path}")
            return output_path

        except RuntimeError as e:
            logger.error(f"TTS generation error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"TTS processing failed: {str(e)}")

# Global instance
try:
    tts_engine = TTSEngine()
except Exception as e:
    logger.critical(f"Failed to initialize TTS system: {str(e)}")
    raise

def text_to_speech(text: str, output_path: str = "static/output.wav") -> Optional[str]:
    """Public interface to the TTS engine"""
    return tts_engine.text_to_speech(text, output_path)