from typing import List


class TurkishDecoder:
    # Define vowel sets as class constants for better performance
    ALL_VOWELS = "aeıioöuüâ"
    INCE_VOWELS = "eiöü"  # Front vowels
    AI_VOWELS = "aıâ"  # Back unrounded
    EI_VOWELS = "ei"  # Front unrounded
    OU_VOWELS = "ou"  # Back rounded
    HARD_CONSONANTS = "fstkçşhp"  # Sert ünsüzler
    WHITESPACE = " \n\t"

    def __init__(self, reverse_dict):
        self.reverse_dict = reverse_dict

    def _tr_capitalize(self, word: str) -> str:
        """Capitalize using Turkish casing rules (i -> İ, ı -> I)."""
        if not word:
            return ""
        if word.startswith("i"):
            return "İ" + word[1:]
        return word.capitalize()

    def _starts_with_vowel(self, word: str) -> bool:
        """Check if word starts with a vowel."""
        return bool(word and word[0] in self.ALL_VOWELS)

    def _ends_with_vowel(self, word: str) -> bool:
        """Check if word ends with a vowel."""
        return bool(word and word[-1] in self.ALL_VOWELS)

    def _ends_with_any(self, word: str, charset: str) -> bool:
        # recursively check until first vowel starts from the end
        i = len(word) - 1
        while i >= 0:
            if word[i] in charset:
                return True
            if word[i] in self.ALL_VOWELS:
                return False
            i -= 1
        return False

    def _ends_with_ince(self, word: str) -> bool:
        """Check if word ends with front vowels (ince ünlü)."""
        if word in ("saat", "kilovatsaat", "ziraat", "itaat", "istikbal"):
            return True
        # check until first vowel recursively
        return self._ends_with_any(word, self.INCE_VOWELS)

    def _ends_with_sert_unsuz(self, word: str) -> bool:
        """Check if word ends with a hard consonant."""
        return bool(word and word[-1] in self.HARD_CONSONANTS)

    def _get_vowel_suffix_index(self, prev_token: str) -> int:
        """Get suffix index based on vowel harmony rules."""
        if self._ends_with_any(prev_token, self.AI_VOWELS):
            return 0
        elif self._ends_with_any(prev_token, self.EI_VOWELS):
            return 1
        elif self._ends_with_any(prev_token, self.OU_VOWELS):
            return 2
        return 3

    def _select_correct_suffix(self, i: int, ids: List[int], prev_token: str) -> str:
        """Select the correct suffix based on morphological rules."""
        suffixes = self.reverse_dict[ids[i]]
        token_id = ids[i]
        # Handle different suffix types with cleaner logic
        if token_id < 20013:
            # Basic suffix selection based on vowel harmony
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

        elif token_id < 20023:  # nın, nin, nun, nün
            return suffixes[self._get_vowel_suffix_index(prev_token)]

        elif token_id == 20023:  # la, le, yla, yle
            end_of_word = True
            if i < len(ids) - 1:
                next_token = self.reverse_dict[ids[i + 1]][0]
                if next_token not in self.WHITESPACE:
                    end_of_word = False
            return self._handle_la_le_suffix(prev_token, suffixes, end_of_word)

        elif token_id <= 20025:  # da, de, ta, te, dan, den, tan, ten
            return self._handle_da_de_suffix(prev_token, suffixes)

        elif 20025 < token_id < 20029:  # dı, di, du, dü, tı, ti, tu, tü, etc.
            return self._handle_di_du_suffix(prev_token, suffixes)

        elif token_id == 20029:  # lık, lik, luk, lük, etc.
            return self._handle_lik_suffix(i, ids, prev_token, suffixes)

        elif token_id == 20030:  # cık, cik, cuk, cük, etc.
            return self._handle_cik_suffix(i, ids, prev_token, suffixes)

        elif token_id == 20031:  # mak, mek, may, mey
            return self._handle_mak_suffix(i, ids, prev_token, suffixes)

        elif token_id == 20032:  # acak, ecek, etc.
            return self._handle_acak_suffix(i, ids, prev_token, suffixes)

        return suffixes[0]

    def _handle_la_le_suffix(
        self, prev_token: str, suffixes: List[str], end_of_word: bool
    ) -> str:
        """Handle la/le/yla/yle suffix selection."""
        if self._ends_with_vowel(prev_token) and end_of_word:
            return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
        else:
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

    def _handle_da_de_suffix(self, prev_token: str, suffixes: List[str]) -> str:
        """Handle da/de/ta/te suffix selection."""
        if self._ends_with_sert_unsuz(prev_token):
            return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
        return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

    def _handle_di_du_suffix(self, prev_token: str, suffixes: List[str]) -> str:
        """Handle dı/di/du/dü suffix selection."""
        base_index = self._get_vowel_suffix_index(prev_token)
        return (
            suffixes[base_index + 4]
            if self._ends_with_sert_unsuz(prev_token)
            else suffixes[base_index]
        )

    def _handle_lik_suffix(
        self, i: int, ids: List[int], prev_token: str, suffixes: List[str]
    ) -> str:
        """Handle lık/lik/luk/lük suffix selection."""
        if i >= len(ids) - 1:
            return suffixes[0]

        next_token = self.reverse_dict[ids[i + 1]][0]
        base_index = self._get_vowel_suffix_index(prev_token)
        return (
            suffixes[base_index + 4]
            if self._starts_with_vowel(next_token)
            else suffixes[base_index]
        )

    def _handle_cik_suffix(
        self, i: int, ids: List[int], prev_token: str, suffixes: List[str]
    ) -> str:
        """Handle cık/cik/cuk/cük suffix selection."""
        if i >= len(ids) - 1:
            return suffixes[0]

        next_token = self.reverse_dict[ids[i + 1]][0]
        base_index = self._get_vowel_suffix_index(prev_token)

        if self._starts_with_vowel(next_token):
            offset = 12 if self._ends_with_sert_unsuz(prev_token) else 8
        else:
            offset = 4 if self._ends_with_sert_unsuz(prev_token) else 0

        return suffixes[base_index + offset]

    def _handle_mak_suffix(
        self, i: int, ids: List[int], prev_token: str, suffixes: List[str]
    ) -> str:
        """Handle mak/mek/may/mey suffix selection."""
        if i >= len(ids) - 1:
            return suffixes[0]

        next_token = self.reverse_dict[ids[i + 1]][0]
        base_index = 1 if self._ends_with_ince(prev_token) else 0
        return (
            suffixes[base_index + 2]
            if self._starts_with_vowel(next_token)
            else suffixes[base_index]
        )

    def _handle_acak_suffix(
        self, i: int, ids: List[int], prev_token: str, suffixes: List[str]
    ) -> str:
        """Handle acak/ecek/yacak/yecek suffix selection."""
        is_vowel_ending = self._ends_with_vowel(prev_token)
        is_ince = self._ends_with_ince(prev_token)

        is_vowel_starting = False
        if i < len(ids) - 1:
            next_token = self.reverse_dict[ids[i + 1]][0]
            is_vowel_starting = self._starts_with_vowel(next_token)

        if is_vowel_starting:
            if is_vowel_ending:
                return suffixes[7] if is_ince else suffixes[6]
            else:
                return suffixes[3] if is_ince else suffixes[2]
        else:
            if is_vowel_ending:
                return suffixes[5] if is_ince else suffixes[4]
            else:
                return suffixes[1] if is_ince else suffixes[0]

    def _select_correct_root(self, i: int, ids: List[int]) -> str:
        """Select the correct root form based on morphological context."""
        token_id = ids[i]
        tokens = self.reverse_dict[token_id]

        if i > len(ids) - 2:
            return tokens[0]

        next_token = self.reverse_dict[ids[i + 1]][0]

        # === EXCEPTIONS: Roots that should NOT soften ===
        # These roots end in consonants that look like they should soften
        # but actually stay unchanged before vowel-initial suffixes
        NO_SOFTENING_ROOTS = {
            204,  # hayat - hayatı (not hayatı -> hayadi)
            220,  # belirt - belirten (not belirden)
            298,  # meslek - mesleki (not mesleği)
        }
        if token_id in NO_SOFTENING_ROOTS:
            return tokens[0]

        # === EXCEPTIONS: Roots where default is variant[1], not variant[0] ===
        # These have multiple forms but the common surface form is the second one
        DEFAULT_VARIANT_1_ROOTS = {
            2227,  # üçlü (not üçle)
            2209,  # yaşı (special handling below)
        }

        # Special case: üçlü - always return üçlü (variant 1) unless specific context
        if token_id == 2227:
            return tokens[1] if len(tokens) > 1 else tokens[0]

        # Akış (aka/akı) Exception (2199) - Default to "akı" (variant 1)
        # "aka" is only used in specific contexts like "akacak"
        if token_id == 2199:
            if i < len(ids) - 1:
                next_str = self.reverse_dict[ids[i + 1]][0]
                # Use "aka" only when followed by vowel-starting suffixes like "acak"
                if next_str.startswith("a") or next_str.startswith("e"):
                    return tokens[0]  # "aka" for "akacak"
            # Default to "akı" for all other cases
            return tokens[1] if len(tokens) > 1 else tokens[0]

        # Ata/Atı Exception (2212) - for "atılırsa", "atılmak", "atıyorlar" etc.
        # Use "atı" (variant 1) when followed by 'l' (passive) or 'y' (yor, yacak)
        if token_id == 2212:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_str = self.reverse_dict[ids[i + 1]][0]
                if next_str.strip().startswith("l") or next_str.strip().startswith("y"):
                    return tokens[
                        1
                    ]  # "atı" + "lırsa" = "atılırsa", "atı" + "yorlar" = "atıyorlar"
            return tokens[0]  # "ata" by default

        # Special case: yaşı/yaşa - return yaşı before 'na' suffix
        if token_id == 2209:
            if i < len(ids) - 1:
                # 20188 = 'na'
                if ids[i + 1] == 20188:
                    return tokens[1] if len(tokens) > 1 else tokens[0]
            return tokens[0]

        # Alın (alın/aln) Exception (182) - Default to "alın" (variant 0)
        # Only use "aln" when followed by possessive vowel suffix
        if token_id == 182:
            if i < len(ids) - 1:
                next_id = ids[i + 1]
                # Only drop vowel for simple possessive suffixes
                # 20034 = 'ı', 20033 = 'i', 20035 = 'u', 20036 = 'ü'
                if next_id in (20034, 20033, 20035, 20036):
                    return tokens[1] if len(tokens) > 1 else tokens[0]  # "aln" + ı
            # Keep "alın" for all other cases
            return tokens[0]

        # Ilim/Ilm Exception (166) - Default to "ilim" (variant 0)
        # Only use "ilm" when followed by single-vowel possessive suffix
        if token_id == 166:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_id = ids[i + 1]
                # Only use "ilm" for possessive/buffer case (ilmi, ilme)
                if next_id in (20033, 20038):  # 'i', 'e'
                    return tokens[1]  # "ilm" + i = "ilmi"
            return tokens[0]  # Default to "ilim"

        # Boya/Boyu Exception (2220) - "boya" (paint) vs "boyu" (height)
        # Use "boyu" (variant 1) by default
        # Use "boya" only for paint-related suffix patterns (boyanan, boyamak, boyalı, etc.)
        if token_id == 2220:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_id = ids[i + 1]
                next_str = self.reverse_dict[next_id][0]
                # Use "boya" only when followed by actual suffix tokens starting with 'n', 'm', 'l', 'd'
                # (boyanan, boyamak, boyalı, boyadan) - these are paint-related contexts
                if (
                    next_id >= 20000
                    and next_str.strip()
                    and next_str.strip()[0] in "nmld"
                ):
                    return tokens[0]  # "boya"
            return tokens[1] if len(tokens) > 1 else tokens[0]  # "boyu" by default

        # Bile/Bili Exception (2307) - for "bilir", "biliyor" vs "biler", "beleyor"
        # Use "bili" (variant 1) when followed by 'r' or 'yor'
        if token_id == 2307:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_str = self.reverse_dict[ids[i + 1]][0]
                if next_str.strip().startswith("r") or next_str.strip() == "yor":
                    return tokens[
                        1
                    ]  # "bili" + "r" = "bilir", "bili" + "yor" = "biliyor"
            return tokens[0]  # Default to "bile"

        # Ada/Adı Exception (2218) - Default to "adı" (variant 1)
        if token_id == 2218:
            if i < len(ids) - 1:
                next_id = ids[i + 1]
                next_str = self.reverse_dict[next_id][0]
                # Use "ada" when followed by 'n' suffixes or 'yı' (for adayı pattern) or 'ma' (adama)
                # 20017 = suffix yı, 32725 = BPE yı, 20002 = ma/me
                if (
                    next_id == 20040
                    or next_str.startswith("n")
                    or next_id in (20017, 32725, 20002, 32763)
                ):
                    return tokens[0]  # "ada" for "adanın", "adayı", "adama"
            # Default to "adı" for most cases
            return tokens[1] if len(tokens) > 1 else tokens[0]

        # Kap/Kab Exception (336) - favor "kapı" (door) over "kab" (container) context
        # "kapımızı" (our door) tokenizes as kap + ımız + ı -> default softens to kabımızı
        # So we prevent softening for 336 in potential door contexts
        if token_id == 336:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_str = self.reverse_dict[ids[i + 1]][0]
                # If followed by vowel (which causes softening default), check if it looks like possessive plural
                # kap + ımız -> kapımız (door) vs kabımız (container)
                # We prioritize "door" (kap) as it's more common
                if self._starts_with_vowel(next_str):
                    return tokens[0]  # Keep "kap"
            return tokens[0]  # Default "kap"

        # Emekli/Emekle Exception (2295) - Default to "emekli" (variant 1)
        if token_id == 2295:
            if i < len(ids) - 1:
                next_id = ids[i + 1]
                # 20041 = 'yor' - for "emekliyor" use base form
                if next_id == 20041:
                    return tokens[0]  # "emekle" + yor = emekliyor
            # Default to "emekli"
            return tokens[1] if len(tokens) > 1 else tokens[0]

        # Tutuk/Tutuğ/Tutk Exception (107) - for "tutkun" (fan/devotee)
        # Use "tutk" (variant 2) when followed by suffix starting with 'u' (for un/unlar etc.)
        # Otherwise use default "tutuk" (don't soften to tutuğ)
        if token_id == 107:
            if len(tokens) > 2 and i < len(ids) - 1:
                next_str = self.reverse_dict[ids[i + 1]][0]
                # Check if next token starts with 'u' (un, unlar, etc.)
                if next_str.strip().startswith("u"):
                    return tokens[2]  # "tutk" + "un" = "tutkun"
            # For other cases, use default form (tutuk), not softened (tutuğ)
            return tokens[0]

        # Başla/Başlı Exception (2206) - for "başlıca" (primary/mainly)
        # Use "başlı" (variant 1) when followed by 'ca/ce' suffix
        if token_id == 2206:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_id = ids[i + 1]
                # 20005 = 'ça/çe' suffix, 20047 = 'ce', 20207 = BPE 'ca'
                if next_id in (20005, 20047, 20207):
                    return tokens[1]  # "başlı" + "ca" = "başlıca"
            # Continue to existing logic below

        # Dip/Dib Exception (2406) - soften to "dib" before vowel suffixes
        # (dibinde, dibini, etc.) Token 2406 is outside the 100-2080 range
        if token_id == 2406:
            if len(tokens) > 1 and i < len(ids) - 1:
                next_str = self.reverse_dict[ids[i + 1]][0]
                if self._starts_with_vowel(next_str.strip()):
                    return tokens[1]  # "dib" + "inde" = "dibinde"
            return tokens[0]  # "dip" by default
        if token_id in [19531, 19968]:  # de, ye
            # Special handling for de/ye narrowing
            # de -> di, ye -> yi when followed by yor or variable suffixes starting with vowel (which get 'y' buffer)
            should_narrow = False

            if next_token.strip() == "yor":
                should_narrow = True
            elif ids[i + 1] in self.reverse_dict:
                # Check if next suffix starts with vowel, invoking 'y' buffer
                # e.g. acak/ecek -> yacak/yecek
                suff_forms = self.reverse_dict[ids[i + 1]]
                if suff_forms and any(
                    s.startswith(("a", "e", "ı", "i", "u", "ü", "o", "ö"))
                    for s in suff_forms
                ):
                    should_narrow = True

            if should_narrow:
                # Replace last char e -> i
                # Handle space prefix
                original = tokens[0]
                if original.endswith("e"):
                    return original[:-1] + "i"
                elif original.endswith("E"):
                    return original[:-1] + "İ"
            return tokens[0]

        if 100 <= token_id < 2080:
            # Skip softening for roots in NO_SOFTENING_ROOTS (already handled above)
            if self._starts_with_vowel(next_token):
                return tokens[1]
            elif token_id <= 110 and next_token.strip() == "ı":
                return tokens[2]
            else:
                return tokens[0]

        elif 2080 <= token_id < 2315:
            if next_token.strip() == "yor":
                return tokens[1]
            else:
                return tokens[0]

        return tokens[0]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs to text."""
        if not ids:
            return ""

        text_parts = []
        i = 0

        while i < len(ids):
            token_id = ids[i]
            # Handle special tokens
            if token_id == 0 and i < len(ids) - 1:  # uppercase
                next_token = self._select_correct_root(i + 1, ids)
                if next_token.startswith(" "):
                    text_parts.append(" " + self._tr_capitalize(next_token.lstrip()))
                else:
                    text_parts.append(self._tr_capitalize(next_token))
                i += 2
                continue
            elif token_id == 1:  # unknown
                text_parts.append("▁u▁")
            elif token_id in self.reverse_dict:
                tokens = self.reverse_dict[token_id]
                if len(tokens) > 1:
                    if token_id < 20000:  # root token
                        text_parts.append(self._select_correct_root(i, ids))
                    else:  # suffix token
                        # Find context from previous tokens
                        # We need enough context for both vowel harmony (looking back past consonants)
                        # and consonant harmony (immediate previous char)
                        prev_token = ""
                        j = len(text_parts) - 1
                        tokens_found = 0

                        # Look back up to 3 tokens or until we have enough context
                        temp_context = []
                        while j >= 0 and tokens_found < 3:
                            part = text_parts[j]
                            temp_context.insert(0, part)
                            if any(c.isalpha() for c in part):
                                tokens_found += 1
                            j -= 1

                        if temp_context:
                            prev_token = "".join(temp_context)

                        text_parts.append(
                            self._select_correct_suffix(i, ids, prev_token)
                        )
                else:
                    text_parts.append(tokens[0])
            else:
                text_parts.append("▁")

            i += 1

        return "".join(text_parts)
