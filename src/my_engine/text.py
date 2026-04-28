import urllib.request
import zipfile
import pathlib
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import urllib.request
import zipfile
import pathlib


def build_glove_embedding_matrix(vocab, glove, embedding_dim=100):
    """Build an embedding matrix aligned to a given vocabulary.

    For each word in the vocabulary, if it exists in the GloVe embeddings, use
    the pretrained GloVe vector. Otherwise, initialize randomly from N(0, 0.1).
    The PAD token (index 0) is always set to the zero vector.

    Args:
        vocab (dict[str, int]): Vocabulary mapping words to integer indices.
        glove (GloVeVocab): Pretrained GloVe word vectors container.
        embedding_dim (int, optional): Dimensionality of the embeddings.
            Defaults to 100.

    Returns:
        torch.Tensor: Embedding matrix of shape (vocab_size, embedding_dim),
            where row i contains the embedding for the word at index i.
    """
    vocab_size = len(vocab)

    # DONE: Initialize an embedding matrix of zeros with shape (vocab_size, embedding_dim).
    #       This will hold the final embeddings for all words in our vocabulary.
    matrix = torch.zeros((vocab_size, embedding_dim), dtype=torch.float32)

    # DONE: Initialize a counter to track how many words we find in GloVe.
    found_in_glove = 0

    # DONE: Loop through each word and its index in the vocabulary.
    for word, idx in vocab.items():
        # DONE: Skip index 0 (the PAD token) — it should remain all zeros.
        if idx == 0:
            continue

        # DONE: Check if the word exists in the GloVe vocabulary (glove.stoi).
        #       If it does, copy the pretrained GloVe vector into our matrix.
        if word in glove.stoi.keys():
            matrix[idx] = glove[word]
            found_in_glove += 1

        # DONE: If the word is not in GloVe, initialize it randomly.
        #       Use a normal distribution N(0, 0.1) by sampling from torch.randn
        #       and scaling by 0.1.
        else:
            matrix[idx] = torch.randn(embedding_dim, dtype=torch.float32) * 0.1

    print(f"GloVe coverage: {found_in_glove}/{vocab_size} words ({100 * found_in_glove / vocab_size:.1f}%)")
    return matrix


class GloVeVocab:
    """A lightweight container for pretrained GloVe word vectors.

    Stores the vocabulary and embedding matrix so that individual word vectors
    can be retrieved by name. The interface intentionally mirrors the legacy
    torchtext.vocab.GloVe API (.stoi, .itos, .vectors, .dim, __getitem__) so
    that downstream code — nearest-neighbor search, analogy arithmetic, and
    embedding matrix construction — works without modification.

    Attributes:
        stoi (dict[str, int]): Maps each word to its row index in `vectors`.
        itos (list[str]):      Maps each row index back to its word.
        vectors (torch.Tensor): Shape (vocab_size, dim). Row i is the embedding
                                for the word itos[i].
        dim (int): Embedding dimensionality (e.g. 50, 100, 200, or 300).
    """

    def __init__(self, words: list, vectors: torch.Tensor):
        """Build lookup structures from an ordered word list and a vector matrix.

        Args:
            words:   Ordered list of vocabulary strings. The position of a word
                     in this list is its row index in `vectors`.
            vectors: Tensor of shape (len(words), dim) containing the embeddings.
        """
        # DONE: Store a copy of the word list as self.itos ("index to string").
        #       This lets us map an integer index back to a word.
        self.itos = words


        # DONE: Build self.stoi ("string to index") as a dict that maps each word
        #       to its integer position in self.itos.
        #       Hint: use enumerate(self.itos) so that idx matches the row in vectors.
        self.stoi = {word:i for i, word in enumerate(self.itos)}


        # DONE: Store the embedding matrix as self.vectors.
        self.vectors = vectors


        # DONE: Store the embedding dimension as self.dim.
        #       Hint: vectors has shape (vocab_size, dim) — use vectors.shape[1].
        self.dim = vectors.shape[1]


    def __len__(self) -> int:
        """Return the vocabulary size (total number of words)."""
        # DONE: Return the number of entries in self.itos.
        return len(self.itos)


    def __contains__(self, word: str) -> bool:
        """Support the `word in glove` membership test."""
        val = self.stoi.get(word, None)
        if val:
            return True

        return False


    def __getitem__(self, word: str) -> torch.Tensor:
        """Return the embedding vector for a word.

        Args:
            word: The word to look up.

        Returns:
            Tensor of shape (dim,) — a 1D vector of floats.

        Raises:
            KeyError: If the word is not in the vocabulary.
        """
        # DONE: Look up the word's index with self.stoi, then return the
        #       corresponding row from self.vectors.
        #       Raise a KeyError with a helpful message if the word is missing.
        index = self.stoi.get(word, None)

        if index is None:
            raise KeyError(f"Word {word} not found in vocabulary")

        return self.vectors[index]




def load_glove_vectors(glove_dir: str = "./data/glove", dim: int = 100) -> GloVeVocab:
    """Download (if needed) and load GloVe 6B word vectors from a plain text file.

    GloVe 6B vectors are distributed by Stanford NLP as a zip archive (~822 MB):
        https://nlp.stanford.edu/data/glove.6B.zip

    The archive contains four text files:
        glove.6B.50d.txt   (50-dimensional vectors)
        glove.6B.100d.txt  (100-dimensional vectors)
        glove.6B.200d.txt  (200-dimensional vectors)
        glove.6B.300d.txt  (300-dimensional vectors)

    Only the file matching `dim` is extracted. Everything is cached under
    `glove_dir` so the download happens only once.

    Text file format — one line per word:
        word  val1  val2  ...  val_dim

    Args:
        glove_dir: Directory where GloVe files will be stored / read from.
                   Defaults to "./data/glove". Created automatically if absent.
        dim:       Embedding dimensionality. One of {50, 100, 200, 300}.

    Returns:
        GloVeVocab instance with .stoi, .itos, .vectors, and .dim populated.
    """
    glove_dir = pathlib.Path(glove_dir)
    glove_dir.mkdir(parents=True, exist_ok=True)

    txt_file = glove_dir / f"glove.6B.{dim}d.txt"
    zip_file = glove_dir / "glove.6B.zip"

    # --- Step 1: Download and extract if the .txt file is not already cached ---
    if not txt_file.exists():
        if not zip_file.exists():
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            print(f"Downloading GloVe 6B (~822 MB) from Stanford NLP...")
            print(f"  Saving to: {zip_file}")
            print("  (This only happens once — the file will be cached.)")

            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                pct = 100 * downloaded / total_size if total_size > 0 else 0
                print(f"  {downloaded/1e6:6.1f} / {total_size/1e6:.1f} MB  ({pct:.1f}%)", end="\r")

            urllib.request.urlretrieve(url, zip_file, reporthook=_progress)
            print("\nDownload complete.")

        target_name = f"glove.6B.{dim}d.txt"
        print(f"Extracting {target_name} from archive...")

        # DONE: Open zip_file as a ZipFile and extract only target_name into glove_dir.
        #       Hint: use `with zipfile.ZipFile(zip_file) as zf:` then zf.extract().
        with zipfile.ZipFile(zip_file) as zf:
            zf.extract(target_name, glove_dir)
        print("Extraction complete.")
    else:
        print(f"Found cached {txt_file.name} — loading...")

    # --- Step 2: Parse the text file into word list and vector matrix ---
    #
    # We read line by line to keep memory usage predictable. Each line is split
    # on whitespace: the first token is the word, the rest are float strings.
    # We accumulate 1D tensors in a list, then stack them into a 2D matrix at
    # the end (one torch.stack call is far faster than repeated concatenation).

    words = []
    raw_vectors = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            # DONE: Strip the trailing newline and split the line on spaces.
            #       tokens[0] is the word; tokens[1:] are the float-string values.
            tokens = line.strip().split(' ')


            # DONE: Convert the numeric string tokens to a 1D float32 tensor.
            #       Hint: torch.tensor([float(v) for v in tokens[1:]], dtype=torch.float32)
            vec = torch.tensor([float(v) for v in tokens[1:]], dtype=torch.float32)


            # DONE: Append `word` to `words` and `vec` to `raw_vectors`.
            words.append(tokens[0])
            raw_vectors.append(vec)



    # DONE: Use torch.stack() to combine the list of 1D tensors into a single
    #       2D tensor of shape (vocab_size, dim). This is more efficient than
    #       building the matrix row by row — torch.stack() allocates one large
    #       contiguous block of memory and fills it in one shot.
    vectors_tensor = torch.stack(raw_vectors, dim=0)

    print(f"Loaded {len(words):,} vectors of dimension {dim}.")
    return GloVeVocab(words, vectors_tensor)


def text_collate_fn(batch, padding_value=0, max_seq_len=None):
    """Collate function for variable-length text sequences.

    Pads all sequences in the batch to the length of the longest sequence,
    using 0 (the PAD index) as the padding value.

    Args:
        batch: List of (token_ids_tensor, label_tensor) tuples from TextDataset.
        padding_value: Index used for padding (default 0, the PAD token).
        max_seq_len: If set, each sequence is truncated to at most this many
            tokens (first ``max_seq_len`` tokens are kept). Use this for
            memory-heavy models such as self-attention, whose memory scales
            as ``O(batch * L^2)`` in sequence length ``L``.

    Returns:
        Tuple of (padded_sequences, labels) where padded_sequences has shape
        (batch_size, max_seq_len) and labels has shape (batch_size,).
    """
    texts, labels = zip(*batch)

    # If max_seq_len is specified, truncate each sequence to this maximum length.
    if max_seq_len is not None:
        texts = tuple(t[:max_seq_len] for t in texts)

    # Pad sequences to the same length (that of the longest sequence in the batch),
    # using padding_value (default is 0 for <PAD> token). Result is (batch_size, max_seq_len).
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=padding_value)

    # Stack the labels into a single tensor of shape (batch_size,).
    labels = torch.stack(labels)

    # Return the padded text sequences and the labels as a tuple.
    return padded_texts, labels

def build_vocab(tokenized_texts: list, max_vocab_size: int = 25000, min_freq: int = 2) -> dict:
    """Build a word-to-index vocabulary from tokenized texts.

    Reserves index 0 for <PAD> and index 1 for <UNK>. Words are ranked by
    frequency and only the top max_vocab_size words with at least min_freq
    occurrences are included.

    Args:
        tokenized_texts: List of lists of tokens (strings).
        max_vocab_size: Maximum number of words in the vocabulary (excluding special tokens).
        min_freq: Minimum frequency a word must have to be included.

    Returns:
        Dictionary mapping words to integer indices.
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            break
        vocab[word] = idx
        idx += 1

    return vocab


