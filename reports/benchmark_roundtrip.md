# Roundtrip Reconstruction Evaluation Report

## Summary

**Exact Match Rate:** 100.0%
**Word-Level Accuracy:** 100.0%

---

## Methodology

Round-trip reconstruction test:
```
Original Text → encode() → token IDs → decode() → Reconstructed Text
```

---

## Detailed Results

| Original | Decoded | Match |
|----------|---------|-------|
| Merhaba dünya | Merhaba dünya | ✓ |
| Türkiye'nin başkenti Ankara'dır | Türkiye'nin başkenti Ankara'dır | ✓ |
| kitaplıktan çıktık | kitaplıktan çıktık | ✓ |
| İstanbul güzel şehir | İstanbul güzel şehir | ✓ |
| BÜYÜK harfler | BÜYÜK harfler | ✓ |
| evlerden kitaplardan | evlerden kitaplardan | ✓ |
| çocuklar okula gitti | çocuklar okula gitti | ✓ |
| Kalktığımızda hep birlikte yürüdük. | Kalktığımızda hep birlikte yürüdük. | ✓ |
| Kitaplarımızdan bazılarını okudum. | Kitaplarımızdan bazılarını okudum. | ✓ |
| Öğretmenlerimize teşekkür ettik. | Öğretmenlerimize teşekkür ettik. | ✓ |
| Bilgisayarlarımızdaki programlar çalışıyor. | Bilgisayarlarımızdaki programlar çalışıyor. | ✓ |
| Üniversitedeki öğrenciler sınava hazırlanıyor. | Üniversitedeki öğrenciler sınava hazırlanıyor. | ✓ |

---

## Error Analysis

### Success Categories
- Simple words with standard morphology: **High accuracy**
- Common suffix combinations: **High accuracy**
- Uppercase handling: **Functional**

### Known Limitations
1. **Complex vowel harmony chains:** Some edge cases in long suffix chains
2. **Exceptional words:** Words with irregular phonological patterns

---

## Implications for Paper

The decoder achieves **100% exact match** on test sentences, demonstrating functional reconstruction for most common patterns. This supports the "near-lossless" characterization in the paper.
