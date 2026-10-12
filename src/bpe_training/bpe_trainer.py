import array
import heapq
import itertools
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import BinaryIO

import regex as re

from src.utils import clock


class MaxHeapItem:
    def __init__(self, item: tuple[int, bytes, bytes]) -> None:
        self.item = item

    def __lt__(self, other) -> bool:
        return self.item > other.item


class WordGroup:
    def __init__(self, word_counts: dict[bytes, int], data_type: str) -> None:
        # use suitable data_type in an array to reduce memory
        match data_type:
            case "B":
                self.UNDEFINED = 255
            case "H":
                self.UNDEFINED = 65535
            case "I":
                self.UNDEFINED = 2 ** 32 - 1

        counts = []  # len: len(word_counts)
        word_edges = [0]  # len: len(word_counts) + 1
        next_ids = []  # len: self.word_edges[-1]
        prev_ids = []  # len: self.word_edges[-1]
        self.bytes = []  # len: self.word_edges[-1]

        # self.bp_ids is {(first bytes, second bytes): {word id in this bag of words: [char id in this word]}}.
        # self.merge runs much faster using this implementation, compared to
        # using [{(first bytes, second bytes): [char id in a word]}]
        self.bp_ids = defaultdict(dict)

        word_i = -1
        for word, count in word_counts.items():
            word_i += 1
            word_l = len(word)

            counts.append(count)
            word_edges.append(word_edges[-1] + word_l)

            # for each word, use array implementation of a linked list
            # combine words together, in order to use array to reduce runtime and memory
            next_ids.extend(list(range(1, word_l)) + [self.UNDEFINED])
            prev_ids.extend([self.UNDEFINED] + list(range(word_l-1)))

            curr_bytes = [bytes([b]) for b in word]
            self.bytes.extend(curr_bytes)

            for i in range(word_l-1):
                self._bp_ids_add((curr_bytes[i], curr_bytes[i+1]), word_i, i)

        self.counts = self._create_compact_uint_array(counts)
        self.word_edges = self._create_compact_uint_array(word_edges)
        self.next_ids = array.array(data_type, next_ids)
        self.prev_ids = array.array(data_type, prev_ids)

    @staticmethod
    def _create_compact_uint_array(lst: list) -> array.array:
        max_entry = max(lst)
        if max_entry <= 255:
            uint_type = "B"
        elif max_entry <= 65535:
            uint_type = "H"
        else:
            uint_type = "I"
        return array.array(uint_type, lst)

    # self.bp_ids[bp][word_id] can be an int or a list of int
    # Always simplify a list of a single int as an int to reduce runtime and memory
    def _bp_ids_add(self, bp: tuple[bytes, bytes], word_id: int, char_id: int) -> None:
        curr_bp_ids = self.bp_ids[bp]
        if word_id not in curr_bp_ids:
            curr_bp_ids[word_id] = char_id
            return

        ids = curr_bp_ids[word_id]
        if isinstance(ids, int):
            curr_bp_ids[word_id] = [ids, char_id]
        else:
            ids.append(char_id)

    def _bp_ids_remove(self, bp: tuple[bytes, bytes], word_id: int, char_id: int) -> None:
        curr_bp_ids = self.bp_ids[bp]
        ids = curr_bp_ids[word_id]
        if isinstance(ids, int):
            del curr_bp_ids[word_id]
            if not curr_bp_ids:
                del self.bp_ids[bp]
        else:
            ids.remove(char_id)
            if len(ids) == 1:
                curr_bp_ids[word_id] = ids[0]

    @property
    def bp_counts(self) -> dict[tuple[bytes, bytes], int]:
        counts = {}
        for bp, curr_bp_ids in self.bp_ids.items():
            for word_i, ids in curr_bp_ids.items():
                count = self.counts[word_i]
                repeat = 1 if isinstance(ids, int) else len(ids)
                counts[bp] = counts.get(bp, 0) + repeat * count
        return counts

    def merge(self, bp: tuple[bytes, bytes]) -> dict[tuple[bytes, bytes], int]:
        if bp not in self.bp_ids:
            return {}

        # Use a plain dict instead of a Counter for bp_diffs to avoid Counter's nonnegative checking
        bp_diffs = {}
        bp_merged = bp[0] + bp[1]

        curr_bp_ids = self.bp_ids.pop(bp)

        for word_i, ids in curr_bp_ids.items():
            ids = [ids] if isinstance(ids, int) else sorted(ids)
            ID_SHIFT = self.word_edges[word_i]
            count = self.counts[word_i]

            for id in ids:
                id_shifted = ID_SHIFT + id
                if self.next_ids[id_shifted] == self.UNDEFINED:
                    continue

                bp_diffs[bp] = bp_diffs.get(bp, 0) - count

                self.bytes[id_shifted] = bp_merged

                next_id = self.next_ids[id_shifted]
                next_id_shifted = ID_SHIFT + next_id

                next_next_id = self.next_ids[next_id_shifted]
                self.next_ids[id_shifted] = next_next_id
                self.next_ids[next_id_shifted] = self.prev_ids[next_id_shifted] = self.UNDEFINED

                if next_next_id != self.UNDEFINED:
                    next_next_id_shifted = ID_SHIFT + next_next_id

                    self.prev_ids[next_next_id_shifted] = id
                    tmp = (bp[1], self.bytes[next_next_id_shifted])
                    bp_diffs[tmp] = bp_diffs.get(tmp, 0) - count

                    # needed if a word includes a pattern like "ttttt"
                    if tmp != bp:
                        self._bp_ids_remove(tmp, word_i, next_id)

                    tmp = (bp_merged, self.bytes[next_next_id_shifted])
                    bp_diffs[tmp] = bp_diffs.get(tmp, 0) + count
                    self._bp_ids_add(tmp, word_i, id)

                prev_id = self.prev_ids[id_shifted]
                if prev_id != self.UNDEFINED:
                    prev_id_shifted = ID_SHIFT + prev_id

                    tmp = (self.bytes[prev_id_shifted], bp[0])
                    bp_diffs[tmp] = bp_diffs.get(tmp, 0) - count
                    self._bp_ids_remove(tmp, word_i, prev_id)

                    tmp = (self.bytes[prev_id_shifted], bp_merged)
                    bp_diffs[tmp] = bp_diffs.get(tmp, 0) + count
                    self._bp_ids_add(tmp, word_i, prev_id)

        return bp_diffs


class BPEAlgorithm:
    def __init__(self, word_counts: dict[bytes, int]) -> None:
        word_counts_B, word_counts_H, word_counts_I = {}, {}, {}
        for word, count in word_counts.items():
            if len(word) <= 255:
                word_counts_B[word] = count
            elif len(word) <= 65535:
                word_counts_H[word] = count
            else:
                word_counts_I[word] = count

        self.word_groups = []
        if word_counts_B:
            self.word_groups.append(WordGroup(word_counts_B, "B"))
        if word_counts_H:
            self.word_groups.append(WordGroup(word_counts_H, "H"))
        if word_counts_I:
            self.word_groups.append(WordGroup(word_counts_I, "I"))

    @clock
    def process(self, num_merges: int) -> list[tuple[bytes, bytes]]:
        bp_counts = {}
        for wg in self.word_groups:
            for bp, count in wg.bp_counts.items():
                bp_counts[bp] = bp_counts.get(bp, 0) + count

        max_heap = [MaxHeapItem((count, b0, b1)) for (b0, b1), count in bp_counts.items()]
        heapq.heapify(max_heap)

        merges = []
        for _ in range(num_merges):
            max_bp = None
            while max_heap:
                count, b0, b1 = heapq.heappop(max_heap).item
                max_bp = (b0, b1)
                if max_bp in bp_counts and bp_counts[max_bp] == count:
                    break

            if max_bp is None:
                raise Exception(f"BPE algorithm stops before {num_merges} merges.")

            merges.append(max_bp)

            for wg in self.word_groups:
                bp_diffs = wg.merge(max_bp)
                for bp, diff in bp_diffs.items():
                    bp_counts[bp] = bp_counts.get(bp, 0) + diff
                    if not bp_counts[bp]:
                        del bp_counts[bp]
                    elif diff != 0:
                        heapq.heappush(max_heap, MaxHeapItem((bp_counts[bp], bp[0], bp[1])))

        return merges


class BPETrainer:
    def __init__(self, n_workers: int, client: ProcessPoolExecutor | None) -> None:
        self.n_workers = n_workers
        self.client = client

    def _find_worker_boundaries(self, file: BinaryIO) -> list[int]:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // self.n_workers
        worker_boundaries = [i * chunk_size for i in range(self.n_workers)] + [file_size]

        mini_chunk_size = 4096

        for i in range(1, self.n_workers):
            pos = worker_boundaries[i]
            file.seek(pos)

            while True:
                mini_chunk = file.read(mini_chunk_size)

                if mini_chunk == b"":
                    worker_boundaries[i] = file_size
                    break

                found_at = mini_chunk.find(b"<|endoftext|>")
                if found_at != -1:
                    worker_boundaries[i] = pos + found_at
                    break

                # -13 in case a mini chunk stops in the middle of <|endoftext|>
                file.seek(-13, os.SEEK_CUR)
                pos += mini_chunk_size - 13

        return sorted(set(worker_boundaries))

    @staticmethod
    @clock
    def _find_word_counts(input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> Counter:
        mini_chunk_size = 4096

        special_tokens_max_len = max(len(t.encode("utf-8")) for t in special_tokens)
        special_tokens_pattern = re.compile("|".join(re.escape(t) for t in special_tokens).encode("utf-8"))

        doc_starts = {start}
        doc_ends = {end}

        word_counts = Counter()
        word_pretoken_pattern = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

        with open(input_path, "rb") as file:
            file.seek(start)
            pos = start

            while True:
                cursor_move = min(mini_chunk_size, end - pos)
                mini_chunk = file.read(cursor_move)

                for match in special_tokens_pattern.finditer(mini_chunk):
                    doc_starts.add(pos + match.end())
                    doc_ends.add(pos + match.start())

                pos += cursor_move
                if pos == end:
                    break

                # in case a mini chunk stops in the middle of a special token
                file.seek(-special_tokens_max_len, os.SEEK_CUR)
                pos -= special_tokens_max_len

            doc_starts = sorted(doc_starts)
            doc_ends = sorted(doc_ends)
            assert len(doc_starts) == len(doc_ends), "Document boundaries unresolved."

            for doc_start, doc_end in zip(doc_starts, doc_ends):
                file.seek(doc_start)
                doc = file.read(doc_end - doc_start).decode("utf-8")

                for match in word_pretoken_pattern.finditer(doc):
                    word_counts[match.group().encode("utf-8")] += 1

        return word_counts

    @clock
    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert "<|endoftext|>" in special_tokens, "<|endoftext|> must be in special_tokens"
        assert vocab_size >= len(special_tokens) + 256, "vocab_size must >= len(special_tokens) + 256"

        futures = []
        word_counts = Counter()

        with open(input_path, "rb") as file:
            bounds = self._find_worker_boundaries(file)

        if self.client is None:
            for start, end in itertools.pairwise(bounds):
                word_counts += self._find_word_counts(input_path, start, end, special_tokens)

        else:
            for start, end in itertools.pairwise(bounds):
                futures.append(self.client.submit(self._find_word_counts, input_path, start, end, special_tokens))

            for future in as_completed(futures):
                word_counts += future.result()

        print(f"Number of pre-BPE words: {len(word_counts)}")
        merges = BPEAlgorithm(word_counts).process(vocab_size - len(special_tokens) - 256)

        vocab = {i: bytes([i]) for i in range(256)}

        for i, (b1, b2) in enumerate(merges, 256):
            vocab[i] = b1 + b2

        for i, t in enumerate(special_tokens, len(vocab)):
            vocab[i] = t.encode("utf-8")

        return vocab, merges
