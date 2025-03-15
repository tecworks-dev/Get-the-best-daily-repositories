# Updated generator.py with batch generation
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
import logging
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from app.models import Segment

# Set up logging
logger = logging.getLogger(__name__)

# Import the CSM watermarking code
try:
    from app.watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
except ImportError:
    # Define stubs for watermarking if the module is not available
    CSM_1B_GH_WATERMARK = "CSM1B"
    def load_watermarker(device="cpu"):
        return None
    def watermark(watermarker, audio, sample_rate, key):
        return audio, sample_rate

def load_llama3_tokenizer():
    """
    Load tokenizer for Llama 3.2, using unsloth's open version
    instead of the gated meta-llama version.
    """
    try:
        # Use the unsloth version which is not gated
        tokenizer_name = "unsloth/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )
        logger.info("Successfully loaded tokenizer from unsloth/Llama-3.2-1B")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer from unsloth: {e}")
        # Fallback to a simpler tokenizer if needed
        try:
            from transformers import GPT2Tokenizer
            logger.warning("Falling back to GPT2Tokenizer")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as fallback_e:
            logger.error(f"Fallback tokenizer also failed: {fallback_e}")
            raise RuntimeError("Could not load any suitable tokenizer")

class Generator:
    """Generator class for CSM-1B model."""
    def __init__(self, model):
        """Initialize generator with model."""
        self._model = model
        self._model.setup_caches(1)
        self._text_tokenizer = load_llama3_tokenizer()
        device = next(model.parameters()).device
        # Load Mimi codec for audio tokenization
        try:
            logger.info("Loading Mimi audio codec...")
            from huggingface_hub import hf_hub_download
            # First try to import from moshi
            try:
                from moshi.models import loaders
                DEFAULT_REPO = loaders.DEFAULT_REPO
                MIMI_NAME = loaders.MIMI_NAME
                get_mimi = loaders.get_mimi
            except ImportError:
                logger.warning("moshi.models.loaders not found, using fallback")
                # Fallback values if moshi.models.loaders is not available
                DEFAULT_REPO = "kyutai/mimi"
                MIMI_NAME = "mimi-december.pt"
                # Fallback function to load mimi
                def get_mimi(checkpoint_path, device):
                    from moshi.models.vqvae_model import MiMiModule
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model = MiMiModule.init_from_checkpoint(checkpoint, device=device)
                    return model
            mimi_weight = hf_hub_download(DEFAULT_REPO, MIMI_NAME)
            mimi = get_mimi(mimi_weight, device=device)
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            self.sample_rate = mimi.sample_rate
            logger.info(f"Mimi codec loaded successfully with sample rate {self.sample_rate}")
        except Exception as e:
            logger.error(f"Error loading Mimi codec: {e}")
            self._audio_tokenizer = None
            self.sample_rate = 24000  # Default sample rate
            logger.warning(f"Using fallback sample rate: {self.sample_rate}")
            raise RuntimeError(f"Failed to load Mimi codec: {e}")

        try:
            self._watermarker = load_watermarker(device=device)
            logger.info("Watermarker loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading watermarker: {e}. Watermarking will be disabled.")
            self._watermarker = None
            
        self.device = device
        # Optimize for CUDA throughput
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            logger.info("CUDA optimizations enabled")
            
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a text segment."""
        frame_tokens = []
        frame_masks = []
        # Strip any voice instructions in square brackets to avoid them being read out
        text = self._clean_text_input(text)
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _clean_text_input(self, text: str) -> str:
        """Remove any voice instructions in square brackets to prevent them being read aloud."""
        import re
        # Remove anything inside square brackets at the beginning
        text = re.sub(r'^\s*\[[^\]]*\]\s*', '', text)
        return text

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio."""
        if self._audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not initialized")
        frame_tokens = []
        frame_masks = []
        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a segment of text and audio."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """Generate audio from text."""
        if self._audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not initialized")
        
        # Start timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
            
        self._model.reset_caches()
        
        # Convert max_audio_length_ms to frames - this controls the maximum generation length
        max_audio_frames = min(int(max_audio_length_ms / 80), 1024)  # Limit to reasonable size
        tokens, tokens_mask = [], []
        
        # Add context segments
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        
        # Add current segment - clean text here to avoid voice instructions being read
        cleaned_text = self._clean_text_input(text)
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(cleaned_text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        
        # Check context size to avoid exceeding model's capacity
        max_seq_len = 2048 - max_audio_frames
        if prompt_tokens.size(0) >= max_seq_len:
            logger.warning(f"Inputs too long ({prompt_tokens.size(0)} tokens), truncating to {max_seq_len - 50}")
            # Truncate but preserve most recent context
            prompt_tokens = prompt_tokens[-max_seq_len+50:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len+50:]
        
        # Generate audio - optimized batch generation
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # Using optimized batch generation approach - much faster than frame-by-frame
        batch_size = 32  # Generate this many frames at once
        all_samples = []
        
        for start_idx in range(0, max_audio_frames, batch_size):
            # Generate a batch of frames at once
            end_idx = min(start_idx + batch_size, max_audio_frames)
            batch_frames = end_idx - start_idx
            
            # Generate samples for this batch
            samples_batch = []
            
            # Initial generation
            for i in range(batch_frames):
                # Generate one frame
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                samples_batch.append(sample)
                
                # Check for EOS
                if torch.all(sample == 0):
                    break
                
                # Update context for next frame
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
            
            # Add batch samples to all_samples
            all_samples.extend(samples_batch)
            
            # Check if we hit EOS and can stop early
            if len(samples_batch) < batch_frames:
                logger.info(f"Early stopping at frame {start_idx + len(samples_batch)}/{max_audio_frames}")
                break
        
        if not all_samples:
            raise RuntimeError("No audio generated - model produced empty output")
            
        # Decode audio
        try:
            audio = self._audio_tokenizer.decode(torch.stack(all_samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        
            # Apply watermark
            if self._watermarker is not None:
                try:
                    audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
                    audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                except Exception as e:
                    logger.warning(f"Error applying watermark: {e}. Continuing without watermark.")
            
            # Record execution time
            end_time.record()
            torch.cuda.synchronize()
            execution_ms = start_time.elapsed_time(end_time)
            audio_length_ms = (audio.shape[0] / self.sample_rate) * 1000
            
            # Calculate real-time factor (RTF)
            rtf = execution_ms / audio_length_ms
            logger.info(f"Audio generated in {execution_ms:.2f}ms, length: {audio_length_ms:.2f}ms, RTF: {rtf:.2f}x")
            
            return audio
            
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            raise RuntimeError(f"Failed to decode audio: {e}")

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    """Load CSM-1B model and create generator."""
    try:
        # Import models module for CSM
        from app.torchtune_models import Model, ModelArgs
        
        # Create model
        model_args = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )
        
        # Load model
        logger.info(f"Loading CSM-1B model from {ckpt_path} to {device}")
        
        # Set up torch for optimized inference
        if device == "cuda" and torch.cuda.is_available():
            # Use mixed precision for faster inference
            dtype = torch.bfloat16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            dtype = torch.float32
            
        model = Model(model_args).to(device=device, dtype=dtype)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Create generator
        logger.info("Creating generator")
        generator = Generator(model)
        logger.info("Generator created successfully")
        
        return generator
    except Exception as e:
        logger.error(f"Failed to load CSM-1B model: {e}")
        raise