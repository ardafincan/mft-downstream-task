"""
Comprehensive tests for TurkishDecoder.

Tests cover:
- Helper functions (vowel detection, consonant rules)
- Suffix selection for all suffix types
- Root selection with consonant softening
- Full decode with various token combinations
- Edge cases and boundary conditions
"""

import json
import pytest
from pathlib import Path


# Load dictionaries once
@pytest.fixture(scope="module")
def vocab_data():
    base_dir = Path(__file__).parent
    with open(base_dir / "ekler.json", "r", encoding="utf-8") as f:
        ekler = json.load(f)
    with open(base_dir / "kokler.json", "r", encoding="utf-8") as f:
        kokler = json.load(f)
    with open(base_dir / "bpe_tokenler.json", "r", encoding="utf-8") as f:
        bpe = json.load(f)

    vocab = {**kokler, **ekler, **bpe}
    reverse_dict = {}
    for key, value in vocab.items():
        if value not in reverse_dict:
            reverse_dict[value] = []
        reverse_dict[value].append(key)

    return {"ekler": ekler, "kokler": kokler, "bpe": bpe, "reverse_dict": reverse_dict}


@pytest.fixture(scope="module")
def decoder(vocab_data):
    from turkish_decoder import TurkishDecoder

    return TurkishDecoder(vocab_data["reverse_dict"])


# =============================================================================
# Tests for _starts_with_vowel
# =============================================================================
class TestStartsWithVowel:
    """Tests for _starts_with_vowel helper function."""

    def test_starts_with_a(self, decoder):
        assert decoder._starts_with_vowel("araba") is True

    def test_starts_with_e(self, decoder):
        assert decoder._starts_with_vowel("ev") is True

    def test_starts_with_i(self, decoder):
        assert decoder._starts_with_vowel("iş") is True
        assert decoder._starts_with_vowel("ışık") is True  # dotless i

    def test_starts_with_o(self, decoder):
        assert decoder._starts_with_vowel("okul") is True
        assert decoder._starts_with_vowel("ördek") is True  # ö

    def test_starts_with_u(self, decoder):
        assert decoder._starts_with_vowel("uzun") is True
        assert decoder._starts_with_vowel("ütü") is True  # ü

    def test_starts_with_consonant(self, decoder):
        assert decoder._starts_with_vowel("kitap") is False
        assert decoder._starts_with_vowel("baba") is False
        assert decoder._starts_with_vowel("çocuk") is False

    def test_empty_string(self, decoder):
        assert decoder._starts_with_vowel("") is False

    def test_single_vowel(self, decoder):
        for v in "aeıioöuü":
            assert decoder._starts_with_vowel(v) is True

    def test_single_consonant(self, decoder):
        for c in "bcdfghjklmnprsştvyz":
            assert decoder._starts_with_vowel(c) is False


# =============================================================================
# Tests for _ends_with_vowel
# =============================================================================
class TestEndsWithVowel:
    """Tests for _ends_with_vowel helper function."""

    def test_ends_with_a(self, decoder):
        assert decoder._ends_with_vowel("araba") is True
        assert decoder._ends_with_vowel("baba") is True

    def test_ends_with_e(self, decoder):
        assert decoder._ends_with_vowel("eve") is True
        assert decoder._ends_with_vowel("gece") is True

    def test_ends_with_i(self, decoder):
        assert decoder._ends_with_vowel("gemi") is True
        assert decoder._ends_with_vowel("taksi") is True

    def test_ends_with_dotless_i(self, decoder):
        assert decoder._ends_with_vowel("sarı") is True
        assert decoder._ends_with_vowel("kırmızı") is True

    def test_ends_with_o(self, decoder):
        assert decoder._ends_with_vowel("radyo") is True

    def test_ends_with_u(self, decoder):
        assert decoder._ends_with_vowel("oku") is True
        assert decoder._ends_with_vowel("türkü") is True  # ü

    def test_ends_with_consonant(self, decoder):
        assert decoder._ends_with_vowel("kitap") is False
        assert decoder._ends_with_vowel("ev") is False
        assert decoder._ends_with_vowel("göz") is False

    def test_empty_string(self, decoder):
        assert decoder._ends_with_vowel("") is False


# =============================================================================
# Tests for _ends_with_ince (front vowel)
# =============================================================================
class TestEndsWithInce:
    """Tests for _ends_with_ince (front vowel harmony detection)."""

    # Front vowels: e, i, ö, ü
    def test_ends_with_e(self, decoder):
        assert decoder._ends_with_ince("ev") is True
        assert decoder._ends_with_ince("gel") is True
        assert decoder._ends_with_ince("gece") is True

    def test_ends_with_i(self, decoder):
        assert decoder._ends_with_ince("gemi") is True
        assert decoder._ends_with_ince("dil") is True
        assert decoder._ends_with_ince("iş") is True

    def test_ends_with_o_umlaut(self, decoder):
        assert decoder._ends_with_ince("göz") is True
        assert decoder._ends_with_ince("kök") is True
        assert decoder._ends_with_ince("ördek") is True

    def test_ends_with_u_umlaut(self, decoder):
        assert decoder._ends_with_ince("gül") is True
        assert decoder._ends_with_ince("düş") is True
        assert decoder._ends_with_ince("türkü") is True

    # Back vowels: a, ı, o, u
    def test_not_ince_with_a(self, decoder):
        assert decoder._ends_with_ince("baba") is False
        assert decoder._ends_with_ince("araba") is False
        assert decoder._ends_with_ince("masa") is False

    def test_not_ince_with_dotless_i(self, decoder):
        assert decoder._ends_with_ince("sarı") is False
        assert decoder._ends_with_ince("kız") is False
        assert decoder._ends_with_ince("balık") is False

    def test_not_ince_with_o(self, decoder):
        assert decoder._ends_with_ince("kol") is False
        assert decoder._ends_with_ince("yol") is False
        assert decoder._ends_with_ince("top") is False

    def test_not_ince_with_u(self, decoder):
        assert decoder._ends_with_ince("uzun") is False
        assert decoder._ends_with_ince("kuş") is False
        assert decoder._ends_with_ince("su") is False

    # Edge cases: consonant endings - check last vowel
    def test_consonant_ending_with_front_vowel(self, decoder):
        assert decoder._ends_with_ince("gel") is True  # last vowel is 'e'
        assert decoder._ends_with_ince("üst") is True  # last vowel is 'ü'
        assert decoder._ends_with_ince("gözlük") is True  # last vowel is 'ü'

    def test_consonant_ending_with_back_vowel(self, decoder):
        assert decoder._ends_with_ince("bak") is False  # last vowel is 'a'
        assert decoder._ends_with_ince("okul") is False  # last vowel is 'u'
        assert decoder._ends_with_ince("kanat") is False

    # Exception words
    def test_exception_words(self, decoder):
        assert decoder._ends_with_ince("saat") is True  # exception
        assert decoder._ends_with_ince("ziraat") is True  # exception

    def test_empty_string(self, decoder):
        assert decoder._ends_with_ince("") is False


# =============================================================================
# Tests for _ends_with_sert_unsuz (hard consonant)
# =============================================================================
class TestEndsWithSertUnsuz:
    """Tests for _ends_with_sert_unsuz (hard consonant detection)."""

    # Hard consonants: f, s, t, k, ç, ş, h, p
    def test_ends_with_p(self, decoder):
        assert decoder._ends_with_sert_unsuz("kitap") is True
        assert decoder._ends_with_sert_unsuz("top") is True

    def test_ends_with_t(self, decoder):
        assert decoder._ends_with_sert_unsuz("at") is True
        assert decoder._ends_with_sert_unsuz("saat") is True

    def test_ends_with_k(self, decoder):
        assert decoder._ends_with_sert_unsuz("ok") is True
        assert decoder._ends_with_sert_unsuz("balık") is True

    def test_ends_with_s(self, decoder):
        assert decoder._ends_with_sert_unsuz("ses") is True
        assert decoder._ends_with_sert_unsuz("arkas") is True

    def test_ends_with_f(self, decoder):
        assert decoder._ends_with_sert_unsuz("elif") is True

    def test_ends_with_h(self, decoder):
        assert decoder._ends_with_sert_unsuz("sabah") is True

    def test_ends_with_ş(self, decoder):
        assert decoder._ends_with_sert_unsuz("iş") is True
        assert decoder._ends_with_sert_unsuz("kuş") is True

    def test_ends_with_ç(self, decoder):
        assert decoder._ends_with_sert_unsuz("saç") is True
        assert decoder._ends_with_sert_unsuz("ağaç") is True

    # Soft consonants should return False
    def test_ends_with_soft_consonant(self, decoder):
        assert decoder._ends_with_sert_unsuz("ev") is False
        assert decoder._ends_with_sert_unsuz("göz") is False
        assert decoder._ends_with_sert_unsuz("yol") is False
        assert decoder._ends_with_sert_unsuz("can") is False

    def test_ends_with_vowel(self, decoder):
        assert decoder._ends_with_sert_unsuz("araba") is False
        assert decoder._ends_with_sert_unsuz("oku") is False

    def test_empty_string(self, decoder):
        assert decoder._ends_with_sert_unsuz("") is False


# =============================================================================
# Tests for _get_vowel_suffix_index
# =============================================================================
class TestGetVowelSuffixIndex:
    """Tests for _get_vowel_suffix_index (4-way vowel harmony)."""

    # Index 0: a, ı (back unrounded) -> ı variant
    def test_back_unrounded_a(self, decoder):
        assert decoder._get_vowel_suffix_index("araba") == 0
        assert decoder._get_vowel_suffix_index("masa") == 0
        assert decoder._get_vowel_suffix_index("bak") == 0

    def test_back_unrounded_dotless_i(self, decoder):
        assert decoder._get_vowel_suffix_index("sarı") == 0
        assert decoder._get_vowel_suffix_index("kız") == 0
        assert decoder._get_vowel_suffix_index("balık") == 0

    # Index 1: e, i (front unrounded) -> i variant
    def test_front_unrounded_e(self, decoder):
        assert decoder._get_vowel_suffix_index("ev") == 1
        assert decoder._get_vowel_suffix_index("gel") == 1
        assert decoder._get_vowel_suffix_index("gece") == 1

    def test_front_unrounded_i(self, decoder):
        assert decoder._get_vowel_suffix_index("gemi") == 1
        assert decoder._get_vowel_suffix_index("dil") == 1
        assert decoder._get_vowel_suffix_index("iş") == 1

    # Index 2: o, u (back rounded) -> u variant
    def test_back_rounded_o(self, decoder):
        assert decoder._get_vowel_suffix_index("kol") == 2
        assert decoder._get_vowel_suffix_index("yol") == 2
        assert decoder._get_vowel_suffix_index("okul") == 2

    def test_back_rounded_u(self, decoder):
        assert decoder._get_vowel_suffix_index("uzun") == 2
        assert decoder._get_vowel_suffix_index("kuş") == 2
        assert decoder._get_vowel_suffix_index("su") == 2

    # Index 3: ö, ü (front rounded) -> ü variant
    def test_front_rounded_o_umlaut(self, decoder):
        assert decoder._get_vowel_suffix_index("göz") == 3
        assert decoder._get_vowel_suffix_index("kök") == 3
        assert decoder._get_vowel_suffix_index("ördek") == 1  # e -> front unrounded

    def test_front_rounded_u_umlaut(self, decoder):
        assert decoder._get_vowel_suffix_index("gül") == 3
        assert decoder._get_vowel_suffix_index("düş") == 3
        assert decoder._get_vowel_suffix_index("türkü") == 3


# =============================================================================
# Tests for suffix selection - ID 20000 (lar/ler)
# =============================================================================
class TestPluralSuffix:
    """Tests for plural suffix selection (lar/ler)."""

    def test_lar_after_back_vowel(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # Back vowel words -> lar
        test_cases = [
            (" araba", "lar"),  # a
            (" balık", "lar"),  # ı
            (" okul", "lar"),  # u
            (" çocuk", "lar"),  # u
            (" masa", "lar"),  # a
        ]

        for root, expected_suffix in test_cases:
            root_id = kokler.get(root)
            if root_id:
                ids = [root_id, ekler["lar"]]
                result = decoder.decode(ids)
                assert expected_suffix in result, f"Failed for {root}: got {result}"

    def test_ler_after_front_vowel(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # Front vowel words -> ler
        test_cases = [
            (" ev", "ler"),  # e
            (" göz", "ler"),  # ö
            (" gül", "ler"),  # ü
            (" dil", "ler"),  # i
        ]

        for root, expected_suffix in test_cases:
            root_id = kokler.get(root)
            if root_id:
                ids = [root_id, ekler["ler"]]
                result = decoder.decode(ids)
                assert expected_suffix in result, f"Failed for {root}: got {result}"


# =============================================================================
# Tests for suffix selection - ID 20024/20025 (da/de/ta/te, dan/den/tan/ten)
# =============================================================================
class TestLocativeAblativeSuffix:
    """Tests for locative (da/de/ta/te) and ablative (dan/den/tan/ten) suffixes."""

    def test_de_after_front_vowel_soft_consonant(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # ev -> evde (front vowel, soft consonant)
        ev_id = kokler.get(" ev")
        if ev_id:
            ids = [ev_id, ekler["de"]]
            result = decoder.decode(ids)
            assert "de" in result and "evde" in result

    def test_da_after_back_vowel_soft_consonant(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # yol -> yolda (back vowel, soft consonant)
        yol_id = kokler.get(" yol")
        if yol_id:
            ids = [yol_id, ekler["da"]]
            result = decoder.decode(ids)
            assert "da" in result

    def test_te_after_hard_consonant_front_vowel(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # ses -> seste (front vowel, hard consonant s)
        ses_id = kokler.get(" ses")
        if ses_id:
            ids = [ses_id, ekler["te"]]
            result = decoder.decode(ids)
            assert "te" in result

    def test_ta_after_hard_consonant_back_vowel(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # kitap -> kitapta (back vowel, hard consonant p)
        kitap_id = kokler.get(" kitap")
        if kitap_id:
            ids = [kitap_id, ekler["ta"]]
            result = decoder.decode(ids)
            assert "ta" in result and "kitapta" in result

    def test_ablative_den_after_front_vowel(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # ev -> evden (front vowel, soft consonant)
        ev_id = kokler.get(" ev")
        if ev_id:
            ids = [ev_id, ekler["den"]]
            result = decoder.decode(ids)
            assert "den" in result and "evden" in result

    def test_ablative_tan_after_hard_consonant(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # kitap -> kitaptan (back vowel, hard consonant)
        kitap_id = kokler.get(" kitap")
        if kitap_id:
            ids = [kitap_id, ekler["tan"]]
            result = decoder.decode(ids)
            assert "tan" in result


# =============================================================================
# Tests for suffix selection - ID 20013-20022 (4-way vowel harmony suffixes)
# =============================================================================
class TestFourWayVowelHarmony:
    """Tests for suffixes with 4-way vowel harmony (ım/im/um/üm etc.)."""

    def test_possessive_im_variants(self, decoder, vocab_data):
        """Test 1st person singular possessive (ım/im/um/üm)."""
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        test_cases = [
            (" baba", "m", "babam"),  # ends with vowel -> just m
            (" ev", "im", "evim"),  # e -> im
            (" göz", "üm", "gözüm"),  # ö -> üm
            (" kol", "um", "kolum"),  # o -> um
            (" balık", "ım", "balığım"),  # ı -> ım (with consonant softening)
        ]

        for root, suffix, expected in test_cases:
            root_id = kokler.get(root)
            suffix_id = ekler.get(suffix)
            if root_id and suffix_id:
                ids = [root_id, suffix_id]
                result = decoder.decode(ids)
                # Check that result ends with the expected suffix
                assert suffix in result or expected in result.replace(" ", "")

    def test_genitive_in_variants(self, decoder, vocab_data):
        """Test genitive suffix (ın/in/un/ün)."""
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        test_cases = [
            (" ev", "in", "evin"),
            (" göz", "ün", "gözün"),
            (" kol", "un", "kolun"),
            (" baba", "nın", "babanın"),  # after vowel -> nın
        ]

        for root, suffix, expected in test_cases:
            root_id = kokler.get(root)
            suffix_id = ekler.get(suffix)
            if root_id and suffix_id:
                ids = [root_id, suffix_id]
                result = decoder.decode(ids)
                assert suffix in result


# =============================================================================
# Tests for suffix selection - ID 20026 (dı/di/du/dü/tı/ti/tu/tü)
# =============================================================================
class TestPastTenseSuffix:
    """Tests for past tense suffix with 8-way variation."""

    def test_di_after_front_vowel_soft_consonant(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        gel_id = kokler.get(" gel")
        if gel_id:
            ids = [gel_id, ekler["di"]]
            result = decoder.decode(ids)
            assert "di" in result

    def test_du_after_back_rounded_soft_consonant(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # Test word ending with back rounded vowel
        oku_id = kokler.get(" oku")
        if oku_id:
            ids = [oku_id, ekler["du"]]
            result = decoder.decode(ids)
            assert "du" in result

    def test_ti_after_hard_consonant(self, decoder, vocab_data):
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # git (ends with t, hard consonant, front vowel)
        git_id = kokler.get(" git")
        if git_id:
            ids = [git_id, ekler["ti"]]
            result = decoder.decode(ids)
            assert "ti" in result


# =============================================================================
# Tests for full decode with complete sentences
# =============================================================================
class TestFullDecode:
    """Tests for full decode functionality."""

    def test_round_trip_simple_words(self, vocab_data):
        """Test encoding and decoding simple words."""
        from turkish_tokenizer import TurkishTokenizer

        tokenizer = TurkishTokenizer()

        words = [
            "evlerden",
            "kitaplardan",
            "gözleri",
            "çocuklar",
            "evde",
            "kitapta",
            "gelmiş",
            "bakmış",
        ]

        for word in words:
            encoded = tokenizer.encode(word)
            decoded = tokenizer.decode(encoded).strip()
            assert (
                word.lower() == decoded.lower()
            ), f"Round-trip failed: {word} -> {decoded}"

    def test_round_trip_with_uppercase(self, vocab_data):
        """Test encoding and decoding with uppercase."""
        from turkish_tokenizer import TurkishTokenizer

        tokenizer = TurkishTokenizer()

        words = [
            "Türkiye",
            "İstanbul",
            "Ankara",
        ]

        for word in words:
            encoded = tokenizer.encode(word)
            decoded = tokenizer.decode(encoded).strip()
            # Case-insensitive comparison since decoder normalizes
            assert (
                word.lower().replace("i̇", "i") in decoded.lower().replace("i̇", "i")
                or decoded.lower() in word.lower()
            )

    def test_empty_ids(self, decoder):
        """Test decoding empty list."""
        assert decoder.decode([]) == ""

    def test_unknown_token(self, decoder):
        """Test handling of unknown token ID."""
        result = decoder.decode([1])  # ID 1 is unknown marker
        assert "▁u▁" in result


# =============================================================================
# Edge cases and boundary tests
# =============================================================================
class TestEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_consecutive_suffixes(self, decoder, vocab_data):
        """Test words with multiple consecutive suffixes."""
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # ev + ler + de + ki
        ev_id = kokler.get(" ev")
        if ev_id and ekler.get("ler") and ekler.get("de") and ekler.get("ki"):
            ids = [ev_id, ekler["ler"], ekler["de"], ekler["ki"]]
            result = decoder.decode(ids)
            assert "evlerdeki" in result.replace(" ", "")

    def test_single_root(self, decoder, vocab_data):
        """Test decoding single root without suffixes."""
        kokler = vocab_data["kokler"]

        ev_id = kokler.get(" ev")
        if ev_id:
            result = decoder.decode([ev_id])
            assert "ev" in result

    def test_suffix_after_suffix(self, decoder, vocab_data):
        """Test vowel harmony carries through suffix chain."""
        ekler = vocab_data["ekler"]
        kokler = vocab_data["kokler"]

        # göz + ler + den (front vowel chain)
        goz_id = kokler.get(" göz")
        if goz_id:
            ids = [goz_id, ekler["ler"], ekler["den"]]
            result = decoder.decode(ids)
            # Should use 'ler' (front) and 'den' (front) based on göz
            assert "ler" in result and "den" in result

    def test_space_prefix_handling(self, decoder, vocab_data):
        """Test that space-prefixed tokens are handled correctly."""
        kokler = vocab_data["kokler"]

        # Multiple words
        ev_id = kokler.get(" ev")
        kitap_id = kokler.get(" kitap")
        if ev_id and kitap_id:
            ids = [ev_id, kitap_id]
            result = decoder.decode(ids)
            # Should have spaces between words
            assert " ev" in result and " kitap" in result


# =============================================================================
# Integration tests with tokenizer
# =============================================================================
class TestTokenizerIntegration:
    """Integration tests with the full tokenizer."""

    def test_long_sentence_round_trip(self):
        """Test round-trip of longer sentences."""
        from turkish_tokenizer import TurkishTokenizer

        tokenizer = TurkishTokenizer()

        sentences = [
            "evlerde kitaplar var",
            "çocuklar okula gitti",
            "göz gözü görmedi",
        ]

        for sentence in sentences:
            encoded = tokenizer.encode(sentence)
            decoded = tokenizer.decode(encoded)
            # Clean and compare
            original_words = sentence.lower().split()
            decoded_words = decoded.strip().lower().split()
            # Check each word is present
            for word in original_words:
                assert any(
                    word in dw or dw in word for dw in decoded_words
                ), f"Word '{word}' not found in decoded: {decoded}"

    def test_regression_sozlerdir(self):
        """Regression test for sözlerdir -> sözlerdür bug."""
        from turkish_tokenizer import TurkishTokenizer

        tokenizer = TurkishTokenizer()
        word = "sözlerdir"
        encoded = tokenizer.encode(word)
        decoded = tokenizer.decode(encoded)
        assert "sözlerdir" in decoded

    def test_regression_capitalization(self):
        """Regression test for Atasözleri capitalization bug."""
        from turkish_tokenizer import TurkishTokenizer

        tokenizer = TurkishTokenizer()
        word = "Atasözleri"
        encoded = tokenizer.encode(word)
        decoded = tokenizer.decode(encoded)
        # Should be capitalized
        assert "Atasözleri" in decoded.replace(" ", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
