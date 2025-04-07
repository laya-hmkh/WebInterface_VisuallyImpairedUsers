import torch
import soundfile as sf
import os
import logging
import re
import numpy as np
from typing import Optional

# Configure logging for tracking progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        """Initialize the TTS engine with Silero model."""
        self.model = None
        self.symbols = None
        self.sample_rate = None
        self.apply_tts = None
        self.max_chunk_size = 140  # Maximum chunk size to avoid model warnings
        self._initialize_model()

    def _initialize_model(self):
        """Load the Silero TTS model and handle initialization errors."""
        try:
            logger.info("Loading Silero TTS model...")
            (self.model, self.symbols, 
             self.sample_rate, _, 
             self.apply_tts) = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='lj_16khz'
            )
            self.model = self.model.to("cpu")  # Force CPU usage    usage for consistency
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Could not initialize TTS engine")

    def _preprocess_text(self, text: str) -> str:
        """Convert numbers and symbols to spoken form for natural TTS output."""
        # Convert numbers to words
        text = re.sub(r'(\d+)', lambda x: self._number_to_words(x.group()), text)
        
        # Replace common symbols with their spoken equivalents
        symbol_map = {'%': 'percent', '$': 'dollars', '#': 'number', '&': 'and', '@': 'at'}
        for sym, word in symbol_map.items():
            text = text.replace(sym, f' {word} ')
        
        # Normalize whitespace
        return ' '.join(text.split())

    def _number_to_words(self, num_str: str) -> str:
        """Convert a number string to its spoken word form (e.g., '123' -> 'one hundred twenty three')."""
        try:
            num = int(num_str)
            if 0 <= num < 20:
                return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
                        'seventeen', 'eighteen', 'nineteen'][num]
            elif 20 <= num < 100:
                tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
                return tens[(num // 10) - 2] + (' ' + self._number_to_words(num % 10) if num % 10 != 0 else '')
            elif 100 <= num < 1000:
                return self._number_to_words(num // 100) + ' hundred' + (
                    ' ' + self._number_to_words(num % 100) if num % 100 != 0 else '')
            else:
                return num_str  # Fallback for unsupported large numbers
        except:
            return num_str  # Return original string if conversion fails

    def _split_text(self, text: str) -> list:
        """Split text into chunks of max_chunk_size or less to respect model limits."""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        words = text.split()
        
        for word in words:
            if len(current_chunk) + len(word) + (1 if current_chunk else 0) <= self.max_chunk_size:
                current_chunk += (word + " ") if not current_chunk else (" " + word)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def text_to_speech(self, text: str, output_path: str = "static/output.wav") -> Optional[str]:
        """
        Convert text to speech and save as a WAV file, handling long inputs by splitting into chunks.

        Args:
            text ( grower): Input text to synthesize (supports numbers and symbols).
            output_path (str): Path to save the output WAV file.

        Returns:
            str or None: Path to the generated audio file, or None if generation fails.

        Raises:
            ValueError: If text is empty or not a string.
            RuntimeError: If audio processing fails.
        """
        # Validate input text
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid text input")
            raise ValueError("Text must be a non-empty string")

        try:
            # Preprocess text for better speech output
            processed_text = self._preprocess_text(text)
            logger.info(f"Processed text for TTS: {processed_text}")

            # Split text into manageable chunks
            text_chunks = self._split_text(processed_text)
            logger.info(f"Text split into {len(text_chunks)} chunks")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Generate and concatenate audio for each chunk
            audio_data = []
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} symbols): {chunk}")
                audio = self.apply_tts(
                    texts=[chunk],
                    model=self.model,
                    sample_rate=self.sample_rate,
                    symbols=self.symbols,
                    device="cpu"
                )[0].numpy()
                audio_data.append(audio)

            # Combine all audio chunks into a single file
            final_audio = np.concatenate(audio_data)
            sf.write(output_path, final_audio, self.sample_rate)
            logger.info(f"Audio saved to {output_path}")
            return output_path

        except RuntimeError as e:
            logger.error(f"TTS generation error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"TTS processing failed: {str(e)}")

# Create a global TTS engine instance
try:
    tts_engine = TTSEngine()
except Exception as e:
    logger.critical(f"Failed to initialize TTS system: {str(e)}")
    raise

def text_to_speech(text: str, output_path: str = "static/output.wav") -> Optional[str]:
    """Public function to access the TTS engine's text-to-speech capability."""
    return tts_engine.text_to_speech(text, output_path)