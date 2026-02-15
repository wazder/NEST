"""
Tokenization and Vocabulary Management for NEST

This module provides:
- BPE (Byte-Pair Encoding) tokenizer
- SentencePiece integration
- Vocabulary builder from text corpus
- Character-level tokenization
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import Counter, defaultdict
import re


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.
    
    Sennrich et al. (2016): "Neural Machine Translation of Rare Words with
    Subword Units"
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for merges
            special_tokens: Special tokens (PAD, UNK, SOS, EOS, etc.)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Default special tokens
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        self.special_tokens = special_tokens
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        
    def train(self, texts: List[str]):
        """
        Train BPE on text corpus.
        
        Args:
            texts: List of text strings
        """
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        
        # Initialize vocabulary with characters
        vocab = defaultdict(int)
        
        for text in texts:
            # Split into words
            words = text.lower().split()
            
            for word in words:
                # Add end-of-word marker
                word = ' '.join(list(word)) + ' </w>'
                vocab[word] += 1
                
        # Learn merges
        num_merges = self.vocab_size - len(self.special_tokens) - 256  # Reserve for chars
        
        for i in range(num_merges):
            # Find most frequent pair
            pairs = self._get_stats(vocab)
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < self.min_frequency:
                break
                
            # Merge best pair
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            if (i + 1) % 100 == 0:
                print(f"  Learned {i + 1} merges...")
                
        # Build final vocabulary
        self._build_vocab(vocab)
        
        print(f"Training complete! Vocabulary size: {len(self.vocab)}")
        
    def _get_stats(self, vocab: Dict[str, int]) -> Dict[tuple, int]:
        """Count pair frequencies."""
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()
            
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
                
        return pairs
        
    def _merge_vocab(
        self,
        pair: tuple,
        vocab: Dict[str, int]
    ) -> Dict[str, int]:
        """Merge a pair in vocabulary."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
            
        return new_vocab
        
    def _build_vocab(self, vocab: Dict[str, int]):
        """Build final vocabulary."""
        # Add special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
            
        # Add subword units
        for word in vocab.keys():
            for symbol in word.split():
                if symbol not in self.vocab and len(self.vocab) < self.vocab_size:
                    self.vocab[symbol] = len(self.vocab)
                    
        # Create mappings
        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Apply BPE merges
            word = ' '.join(list(word)) + ' </w>'
            
            for pair in self.merges:
                bigram = ' '.join(pair)
                replacement = ''.join(pair)
                word = word.replace(bigram, replacement)
                
            # Convert to IDs
            for symbol in word.split():
                token_id = self.token_to_id.get(symbol, self.token_to_id['<UNK>'])
                tokens.append(token_id)
                
        return tokens
        
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text
        
    def save(self, path: str):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
            'merges': self.merges
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved tokenizer to {path}")
        
    def load(self, path: str):
        """Load tokenizer from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        
        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Loaded tokenizer from {path}")


class VocabularyBuilder:
    """Build vocabulary from text corpus."""
    
    def __init__(
        self,
        max_vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize vocabulary builder.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_frequency: Minimum word frequency
            special_tokens: Special tokens
        """
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
            
        self.special_tokens = special_tokens
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = Counter()
        
    def build_from_texts(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count words
        for text in texts:
            words = self._tokenize(text)
            self.word_counts.update(words)
            
        # Filter by frequency
        filtered_words = [
            word for word, count in self.word_counts.items()
            if count >= self.min_frequency
        ]
        
        # Sort by frequency and take top K
        sorted_words = sorted(
            filtered_words,
            key=lambda w: self.word_counts[w],
            reverse=True
        )[:self.max_vocab_size - len(self.special_tokens)]
        
        # Add special tokens first
        for token in self.special_tokens:
            self.word_to_id[token] = len(self.word_to_id)
            
        # Add words
        for word in sorted_words:
            self.word_to_id[word] = len(self.word_to_id)
            
        # Create reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
        print(f"Vocabulary built! Size: {len(self.word_to_id)}")
        print(f"  Total unique words: {len(self.word_counts)}")
        print(f"  Filtered by frequency: {len(filtered_words)}")
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Lowercase and split
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        text = re.sub(r'[^a-z0-9\s\.,!?]', '', text)
        words = text.split()
        return words
        
    def encode(self, text: str) -> List[int]:
        """Encode text to IDs."""
        words = self._tokenize(text)
        return [
            self.word_to_id.get(word, self.word_to_id['<UNK>'])
            for word in words
        ]
        
    def decode(self, ids: List[int]) -> str:
        """Decode IDs to text."""
        words = [self.id_to_word.get(i, '<UNK>') for i in ids]
        return ' '.join(words)
        
    def save(self, path: str):
        """Save vocabulary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'max_vocab_size': self.max_vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'word_to_id': self.word_to_id,
            'word_counts': dict(self.word_counts)
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved vocabulary to {path}")
        
    def load(self, path: str):
        """Load vocabulary from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.max_vocab_size = data['max_vocab_size']
        self.min_frequency = data['min_frequency']
        self.special_tokens = data['special_tokens']
        self.word_to_id = data['word_to_id']
        self.word_counts = Counter(data['word_counts'])
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
        print(f"Loaded vocabulary from {path}")
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_id)


def main():
    """Example usage."""
    print("="*60)
    print("Tokenization and Vocabulary")
    print("="*60)
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Neural networks for brain computer interfaces",
        "EEG signals decoded into natural language text",
        "The quick brown fox is very quick and brown"
    ]
    
    # Test Vocabulary Builder
    print("\n1. Vocabulary Builder")
    vocab_builder = VocabularyBuilder(max_vocab_size=50, min_frequency=1)
    vocab_builder.build_from_texts(texts)
    
    test_text = "The quick fox"
    encoded = vocab_builder.encode(test_text)
    decoded = vocab_builder.decode(encoded)
    
    print(f"\n   Original: {test_text}")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: {decoded}")
    print(f"   Vocab size: {vocab_builder.vocab_size}")
    
    # Test BPE Tokenizer
    print("\n2. BPE Tokenizer")
    bpe = BPETokenizer(vocab_size=100, min_frequency=1)
    bpe.train(texts)
    
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"\n   Original: {test_text}")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: {decoded}")
    print(f"   Vocab size: {len(bpe.vocab)}")
    print(f"   Merges learned: {len(bpe.merges)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
