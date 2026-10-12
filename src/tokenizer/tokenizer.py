import heapq
import pickle
from collections import OrderedDict
from collections.abc import Iterable, Iterator

import regex as re


class LRUDict:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache = OrderedDict()

    def __getitem__(self, word: str) -> list[int]:
        if word not in self.cache:
            raise Exception(f"{word} not in this LRUDict.")

        self.cache.move_to_end(word)
        return self.cache[word]

    def __setitem__(self, word: str, ids: list[int]) -> None:
        if word in self.cache:
            self.cache[word] = ids
            self.cache.move_to_end(word)
        else:
            self.cache[word] = ids
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def __contains__(self, word: str) -> bool:
        return word in self.cache


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        lru_cache_capacity: int = 0,
    ) -> None:
        self.id_to_token = vocab
        self.token_to_id = {token: i for i, token in vocab.items()}

        self.merges = merges
        self.bp_to_id = {bp: i for i, bp in enumerate(merges)}

        eod_of_text_token = "<|endoftext|>"
        self.eod_of_text_token_id = self.token_to_id[eod_of_text_token.encode("utf-8")]
        self.special_token_ids = {eod_of_text_token: self.eod_of_text_token_id}
        _special_tokens = {eod_of_text_token}

        if special_tokens is not None:
            _special_tokens = set(special_tokens)

            vocab_size = len(vocab)
            for t in _special_tokens:
                if t == eod_of_text_token:
                    continue
                vocab[vocab_size] = t.encode("utf-8")
                self.special_token_ids[t] = vocab_size
                vocab_size += 1

            _special_tokens.add(eod_of_text_token)

        self.special_tokens_pattern = re.compile("|".join(re.escape(t) for t in sorted(_special_tokens, key=len, reverse=True)))
        self.word_pretoken_pattern = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

        self.cache = LRUDict(lru_cache_capacity) if lru_cache_capacity > 0 else None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
        lru_cache_capacity: int = 0,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return Tokenizer(vocab, merges, special_tokens=special_tokens, lru_cache_capacity=lru_cache_capacity)

    def _encode_one_word(self, word: str) -> list[int]:
        if self.cache is not None and word in self.cache:
            return self.cache[word]

        word_bytes = [bytes([b]) for b in word.encode("utf-8")]
        word_l = len(word_bytes)
        next_ids = list(range(1, word_l)) + [-1]
        prev_ids = [-1] + list(range(word_l-1))

        min_heap = []
        for i in range(word_l-1):
            bp = (word_bytes[i], word_bytes[i+1])
            if bp not in self.bp_to_id:
                continue

            min_heap.append((self.bp_to_id[bp], i))

        heapq.heapify(min_heap)

        while min_heap:
            bp, id = heapq.heappop(min_heap)
            bp = self.merges[bp]
            next_id = next_ids[id]

            if bp != (word_bytes[id], word_bytes[next_id]):
                continue

            bp_merged = bp[0] + bp[1]
            word_bytes[id] = bp_merged

            next_next_id = next_ids[next_id]
            next_ids[id] = next_next_id
            next_ids[next_id] = prev_ids[next_id] = -1

            if next_next_id != -1:
                prev_ids[next_next_id] = id

                tmp = (bp_merged, word_bytes[next_next_id])
                if tmp in self.bp_to_id:
                    heapq.heappush(min_heap, (self.bp_to_id[tmp], id))

            prev_id = prev_ids[id]
            if prev_id != -1:
                tmp = (word_bytes[prev_id], bp_merged)

                if tmp in self.bp_to_id:
                    heapq.heappush(min_heap, (self.bp_to_id[tmp], prev_id))

        i = 0
        token_ids = []
        while i >= 0:
            token_ids.append(self.token_to_id[word_bytes[i]])
            i = next_ids[i]

        if self.cache is not None:
            self.cache[word]= token_ids

        return token_ids

    def _encode_one_chunk(self, chunk: str) -> list[int]:
        return sum((self._encode_one_word(match.group()) for match in self.word_pretoken_pattern.finditer(chunk)), [])

    def encode(self, text: str) -> list[int]:
        chunks = re.split(f"({self.special_tokens_pattern.pattern})", text)
        ids = []
        for i, chunk in enumerate(chunks):
            if i & 1:
                ids.append(self.special_token_ids[chunk])
            elif chunk:
                ids.extend(self._encode_one_chunk(chunk))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_token[i] for i in ids).decode("utf-8", errors="replace")

    def is_end_of_text(self, id: int) -> bool:
        return id == self.eod_of_text_token_id
