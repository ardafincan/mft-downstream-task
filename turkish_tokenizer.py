import json
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

from turkish_decoder import TurkishDecoder


class TokenType(Enum):
    ROOT = "ROOT"
    SUFFIX = "SUFFIX"
    BPE = "BPE"


class TurkishTokenizer:
    def __init__(self):
        # Get the directory where this module is located
        package_dir = os.path.dirname(os.path.abspath(__file__))

        # Load JSON files from the package directory
        with open(
            os.path.join(
                package_dir,
                "mft_rust/src/resources/kokler.json",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            roots = json.load(f)
        with open(
            os.path.join(package_dir, "mft_rust/src/resources/ekler.json"),
            "r",
            encoding="utf-8",
        ) as f:
            suffixes = json.load(f)
        with open(
            os.path.join(package_dir, "mft_rust/src/resources/bpe_tokenler.json"),
            "r",
            encoding="utf-8",
        ) as f:
            bpe_tokens = json.load(f)

        # Store the dictionaries as instance attributes
        self.roots = roots
        self.suffixes = suffixes
        self.bpe_tokens = bpe_tokens

        # Now create vocab and reverse dict
        self.vocab = self.get_vocab()
        self.reverse_dict = {}

        # Helper to populate reverse dict
        def add_to_reverse(source_dict):
            for key, value in source_dict.items():
                if value not in self.reverse_dict:
                    self.reverse_dict[value] = []
                # Avoid duplicates
                if key not in self.reverse_dict[value]:
                    self.reverse_dict[value].append(key)

        add_to_reverse(self.roots)
        add_to_reverse(self.suffixes)
        add_to_reverse(self.bpe_tokens)

        self.decoder = TurkishDecoder(self.reverse_dict)

        self.vocab_size = len(self.reverse_dict)

        self.max_root_len = max(len(k) for k in roots) if roots else 0
        self.max_suffix_len = max(len(k) for k in suffixes) if suffixes else 0
        self.max_bpe_len = max(len(k) for k in bpe_tokens) if bpe_tokens else 0

        self.uppercase_marker = {
            "token": "<uppercase>",
            "id": roots["<uppercase>"],
            "type": TokenType.ROOT,
        }
        self.unknown_marker = {
            "token": "<unknown>",
            "id": roots["<unknown>"],
            "type": TokenType.ROOT,
        }
        self.space_marker = {"token": " ", "id": roots[" "], "type": TokenType.ROOT}

        # added to be compatible with SFTTrainer
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = roots[self.pad_token]
        self.eos_token_id = roots[self.eos_token]

    # added to be compatible with SFTTrainer
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]

    # added to be compatible with SFTTrainer
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.reverse_dict[id] for id in ids]

    def get_vocab(self) -> Dict[str, int]:
        return {**self.roots, **self.suffixes, **self.bpe_tokens}

    def _tokenize_word(self, word: str) -> Tuple[List[dict], List[int]]:
        uppercase_indices = [i for i, c in enumerate(word) if c.isupper()]
        result = []

        segments = self._camel_split_with_positions(word)

        for seg, orig_pos in segments:
            if orig_pos < len(word) and word[orig_pos].isupper():
                result.append(self.uppercase_marker)

                # Only prepend space if at start and not whitespace
                should_add_space = orig_pos == 0 and not seg.isspace()

                if should_add_space:
                    seg = " " + seg

            s = self._tr_lower(seg)
            pos = 0

            while pos < len(s):
                substr = s[pos:]

                r_matches = self._all_prefix_matches(
                    substr, self.roots, self.max_root_len
                )
                b_matches = self._all_prefix_matches(
                    substr, self.bpe_tokens, self.max_bpe_len
                )
                s_matches = self._all_prefix_matches(
                    substr, self.suffixes, self.max_suffix_len
                )

                candidates = []
                for r_id, r_tok in r_matches:
                    candidates.append(("ROOT", r_tok, r_id, len(r_tok), TokenType.ROOT))
                for b_id, b_tok in b_matches:
                    candidates.append(("BPE", b_tok, b_id, len(b_tok), TokenType.BPE))
                for s_id, s_tok in s_matches:
                    candidates.append(
                        ("SUFFIX", s_tok, s_id, len(s_tok), TokenType.SUFFIX)
                    )

                if not candidates:
                    result.append(self.unknown_marker)
                    pos += 1
                    continue

                best_candidate = None
                best_score = -1

                for c_type, c_tok, c_id, c_len, c_enum in candidates:
                    score = c_len
                    remainder = substr[c_len:]

                    if not remainder:
                        # Full match bonus
                        score += 5
                    else:
                        # Follow-up suffix bonus
                        s_next_id, s_next_tok = self._longest_prefix_lookup(
                            remainder, self.suffixes, self.max_suffix_len
                        )
                        if s_next_id is not None:
                            # Ignore 1-char variants to prefer atomic roots (e.g. Kapı vs Kap+ı)
                            if len(s_next_tok) > 1:
                                score += len(s_next_tok)

                    if score > best_score:
                        best_score = score
                        best_candidate = (c_tok, c_id, c_enum)
                    elif score == best_score:
                        # Tie-break Priority: Root > BPE > Suffix
                        if (
                            c_enum == TokenType.ROOT
                            and best_candidate[2] != TokenType.ROOT
                        ):
                            best_candidate = (c_tok, c_id, c_enum)
                        elif (
                            c_enum == TokenType.BPE
                            and best_candidate[2] == TokenType.SUFFIX
                        ):
                            best_candidate = (c_tok, c_id, c_enum)

                result.append(
                    {
                        "token": best_candidate[0],
                        "id": best_candidate[1],
                        "type": best_candidate[2],
                    }
                )
                pos += len(best_candidate[0])
                continue

                result.append(self.unknown_marker)
                pos += 1

        return result, uppercase_indices

    def tokenize_text(self, text: str) -> Tuple[List[dict], List[int]]:
        final_tokens = []
        uppercase_indices = [i for i, c in enumerate(text) if c.isupper()]

        parts = text.split(" ")
        for idx, part in enumerate(parts):
            part = part.strip()
            part = " " + part
            if part.strip():
                tokens, _ = self._tokenize_word(part)

                cleaned_tokens = []
                for i, token in enumerate(tokens):

                    if (
                        i >= 2
                        and not (0 <= token["id"] <= 19999)
                        and tokens[i - 2] == self.uppercase_marker
                        and tokens[i - 1] == self.space_marker
                    ):
                        cleaned_tokens.pop(-1)

                    # If this token is uppercase_marker, check previous token
                    if (
                        token == self.uppercase_marker
                        and len(cleaned_tokens) > 0
                        and cleaned_tokens[-1] == self.space_marker
                    ):
                        should_pop = False
                        if i + 1 < len(tokens):
                            next_tok_str = tokens[i + 1]["token"]
                            if next_tok_str.startswith(" "):
                                should_pop = True

                        if should_pop:
                            cleaned_tokens.pop()  # remove the last " " before uppercase
                    cleaned_tokens.append(token)

                final_tokens.extend(cleaned_tokens)

        return final_tokens, uppercase_indices

    def encode(self, text: str) -> List[int]:
        tokens, _ = self.tokenize_text(text)
        return [t["id"] for t in tokens]

    def tokenize(self, text: str) -> List[str]:
        tokens, _ = self.tokenize_text(text)
        return [t["token"] for t in tokens]

    def _longest_prefix_lookup(
        self, s: str, table: Dict[str, int], max_len: int = None
    ) -> Tuple[Optional[int], str]:
        end = min(len(s), max_len) if max_len else len(s)
        for i in range(end, 0, -1):
            cand = s[:i]
            if cand in table:
                return table[cand], cand
        return None, ""

    def _all_prefix_matches(
        self, s: str, table: Dict[str, int], max_len: int = None
    ) -> List[Tuple[int, str]]:
        matches = []
        end = min(len(s), max_len) if max_len else len(s)
        for i in range(end, 0, -1):
            prefix = s[:i]
            if prefix in table:
                matches.append((table[prefix], prefix))
        return matches

    def _tr_lower(self, word: str) -> str:
        if "I" in word or "İ" in word:
            word = word.replace("İ", "i").replace("I", "ı")
        return word.lower()

    def _camel_split_with_positions(self, word: str) -> List[Tuple[str, int]]:
        if not word:
            return []

        parts = []
        start = 0

        for i in range(1, len(word)):
            if word[i].isupper():
                if start < i:
                    parts.append((self._tr_lower(word[start:i]), start))
                start = i

        if start < len(word):
            parts.append((self._tr_lower(word[start:]), start))

        return parts if parts else [(self._tr_lower(word), 0)]

    def decode(self, ids: List[int]) -> str:
        return self.decoder.decode(ids)

    # added to be compatible with SFTTrainer
    def __call__(self, text: str) -> Dict[str, List[int]]:
        input_ids = self.encode(text)
        attention_mask = [1 for _ in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}
